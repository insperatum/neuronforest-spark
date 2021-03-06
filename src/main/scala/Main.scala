import java.io
import java.io._
import java.text.SimpleDateFormat
import java.util.{Date, Calendar}

import main.scala.org.apache.spark.mllib.tree.model.MyModel
import org.apache.spark.mllib.tree.impl.MyTreePoint
import org.apache.spark.mllib.tree.model._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.loss.MalisLoss
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.configuration._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.tree.impurity._
import org.apache.spark.mllib.tree.configuration.QuantileStrategy._
import org.apache.spark.mllib.tree.configuration.Algo._

object Main {

  def main(args:Array[String]): Unit = {
    try {
      mainbody(args)
    } catch {
      case e:Throwable =>
        println("Error:\n" + e.toString)
        e.printStackTrace()
        throw e
    }
  }

  def mainbody(args:Array[String]) {
    val start_time = System.currentTimeMillis()

    //------------------------ Init ---------------------------------------
    val s = getSettingsFromArgs(args)
    println("Settings:\n" + s.toVerboseString)

    //val offsets = for (x <- s.dimOffsets; y <- s.dimOffsets) yield (x, y)
    val nFeatures = s.featureSettings.baseFeaturesAndOffsets.size
    val conf = new SparkConf().setAppName("Hello").set("spark.shuffle.spill", "false").set("spark.logConf", "true")
    if (!s.master.isEmpty) conf.setMaster(s.master)
    val sc = new SparkContext(conf)

    val timestr = new SimpleDateFormat("yyyy-MM-dd HH-mm-ss").format(new Date())
    val save_to = s.save_to + "/" + s.name + "-" + timestr

//    //-------------------------- Train -------------------------------------
    val (splits, bins) = NeuronUtils.getSplitsAndBins(s.subvolumes.train, s.featureSettings.nBaseFeatures, s.data_root, s.maxBins, s.featureSettings.baseFeaturesAndOffsets)
    val (train, dimensions_train) = NeuronUtils.loadData(sc, s.numExecutors, s.subvolumes.train, s.featureSettings.nBaseFeatures,
      s.data_root, s.maxBins, s.featureSettings.baseFeaturesAndOffsets, 1, bins, fromFront = true)
    //train.persist(StorageLevel.MEMORY_ONLY_SER)
    val strategy = new MyStrategy(Regression, MyVariance, s.maxDepth, 2, s.maxBins, Sort, Map[Int, Int](),
      maxMemoryInMB = s.maxMemoryInMB, useNodeIdCache = s.useNodeIdCache, subsamplingRate = s.bagging, minInstancesPerNode = s.leafSize)

    val model: MyEnsembleModelNew[_] = if (s.iterations == 0) {
      s.initialModel match {
        case m:InitialTrainModel =>
          println("Training a Random Forest Model (no gradient boosting)")
          //    Random Forest
          MyRandomForest.trainSerial(train, strategy, m.initialTrees, s.featureSettings.featureSubsetStrategy: String, s.seed,
            nFeatures, dimensions_train.map(_.map(_.n_targets).sum).sum, splits, bins)
        case m:InitialLoadedModel => m.load()
      }

    } else {
      println("Training a Gradient boosted model")
      //    Gradient Boosting
      val boostingStrategy = new MyBoostingStrategy(strategy, MalisLoss, s.initialModel, s.iterations,
        s.malisSettings.treesPerIteration, s.malisSettings.learningRate, s.malisSettings.momentum)
      //    val (model, grads, seg) = new MyGradientBoostedTrees(boostingStrategy).run(train, boostingStrategy, nFeatures,
      //      dimensions_train.map(_.n_targets).sum, splits, bins, s.featureSubsetStrategy)
      new MyGradientBoostedTrees(boostingStrategy).run(train, boostingStrategy, nFeatures,
        dimensions_train.map(_.map(_.n_targets).sum).sum, splits, bins, s.malisSettings.subsampleProportion, s.featureSettings.featureSubsetStrategy, save_to + "/malis", s.saveGradients, s.seed)

    }


    val training_time = (System.currentTimeMillis() - start_time)/60000
    println("Trained (took " + training_time + " minutes.")

    //-------------------------- Train 1D! -------------------------------------
//    val (train, dimensions_train) = NeuronUtils.randomLabeledData1D(sc, s.subvolumes, s.nBaseFeatures, s.data_root, s.trainFraction, fromFront = true)
//    //train.persist(StorageLevel.MEMORY_ONLY_SER)
//    val strategy = new Strategy(Regression, Variance, s.maxDepth, 2, s.maxBins, Sort, Map[Int, Int](), maxMemoryInMB = s.maxMemoryInMB, useNodeIdCache = s.useNodeIdCache)
//
//    val model: RandomForestModel = if (s.iterations == 0) {
//      println("Training a Random Forest Model (no gradient boosting)")
//      //    Random Forest
//      RandomForest.trainRegressor(train, strategy, s.initialTrees, s.featureSubsetStrategy, 1)
//    } else {
//      println("Training a Gradient boosted model")
//      //    Gradient Boosting
//      ???
//    } /*else {
//      println(s.mode + " is not a valid mode!")
//      ???
//    }*/
//    println("trained.")



    //-------------------------- Save Model ---------------------------------
    if(s.save_model_to != "") {
      val model_dir = s.save_model_to + "/" + s.name + "-" + timestr
      println("\nSaving model to " + model_dir)
      new io.File(model_dir).mkdirs()

      val fos = new FileOutputStream(model_dir + "/model.txt")
      val oos = new ObjectOutputStream(fos)
      oos.writeObject(model)
      oos.close()

      val fwdescription = new FileWriter(model_dir + "/description.txt", false)
      fwdescription.write(s.toVerboseString)
      fwdescription.write("\nTraining took " + training_time + " minutes.")
      fwdescription.close()
      println("Saved")
    }
    //-------------------------- Test ---------------------------------------
    val (test, dimensions_test) = NeuronUtils.loadData(sc, s.numExecutors, s.subvolumes.test, s.featureSettings.nBaseFeatures, s.data_root,
      s.maxBins, s.featureSettings.baseFeaturesAndOffsets, 1, bins, fromFront = false)

//    val allPartialModels:Seq[MyEnsembleModelNew[_]] = model.getPartialModels

//    println("There are " + allPartialModels.size + " partial models")

    val (train_cached, _) = NeuronUtils.cached(train)
    val (test_cached, _) = NeuronUtils.cached(test)

    //val testPartialModels = (s.testPartialModels :+ allPartialModels.size).distinct //ensure that the final model is tested
    val testPartialModels = if(s.testPartialModels.isEmpty) Seq(model.nElems)
                            else if(s.testPartialModels.head == -1) Seq()
                            else s.testPartialModels
    val testDepths = if(s.testDepths.isEmpty) Seq(Integer.MAX_VALUE) else s.testDepths

    val partialSegments = model.getPartialSegments(testPartialModels)

    //var j = 0
    //while (j < testDepths.size) { // WHYYYY???
      //val depth = testDepths(j)
      //println("Depth " + depth)
      //j += 1

/*
      def capDepth(m: Any):Unit = {
        model.elems.head match {
          case _: MyDecisionTreeModel =>
            partialSegments.foreach(_.elems.foreach(_.asInstanceOf[MyDecisionTreeModel].capDepth(depth)))
          //          allPartialModels.foreach(_.elems.foreach(_.asInstanceOf[MyDecisionTreeModel].capDepth(depth)))
          case _: MyEnsembleModel[_] =>
            partialSegments.foreach(_.elems.foreach(x => capDepth(x)))
          //          allPartialModels.foreach(_.elems.foreach(_.asInstanceOf[MyRandomForestModelNew].trees.foreach(_.capDepth(depth))))
        }
      }*/


//      model.elems.head match {
//        case _: MyDecisionTreeModel =>
//          partialSegments.foreach(_.elems.foreach(_.asInstanceOf[MyDecisionTreeModel].capDepth(depth)))
////          allPartialModels.foreach(_.elems.foreach(_.asInstanceOf[MyDecisionTreeModel].capDepth(depth)))
//        case _: MyRandomForestModelNew =>
//          partialSegments.foreach(_.elems.foreach(_.asInstanceOf[MyRandomForestModelNew].trees.foreach(_.capDepth(depth))))
////          allPartialModels.foreach(_.elems.foreach(_.asInstanceOf[MyRandomForestModelNew].trees.foreach(_.capDepth(depth))))
//      }


      var i=0
      var nElems = 0
      var weightSum = 0
      var currentPredictionsTrain = train_cached.map { point => DoubleTuple.Zero }
      var currentPredictionsTest = test_cached.map { point => DoubleTuple.Zero }
      while(i < testPartialModels.size) { //shit goes weird when I use a for loop. WHY? todo:INVESTIGATE
        val partialSegment = partialSegments(i)
//        val partialModel = allPartialModels(testPartialModels(i) - 1)
        i+=1
        nElems = nElems + partialSegment.nElems

        println("Broadcasting partial model with " + nElems + " elements"/*, at depth " + depth*/)
        println("(partialSegment.nElems = " + partialSegment.nElems + ")")
        val mBroadcasts = partialSegment.elems.map(m => sc.broadcast(m))

        println("Predicting on training data")
        val sumType = if(partialSegment.isSum) "sum" else if(nElems == partialSegment.nElems) "only" else "average"
        val partialnElems = partialSegment.nElems
        // Training Error
        val trainPartialPredictions/*, trainPartialUnpersist*/ = MyTreeEnsembleModel.predictWithBroadcasts(mBroadcasts, partialSegment, train_cached)
        //val trainPartialPredictions/*, trainPartialUnpersist*/ = MyTreeEnsembleModel.predictSerial(partialSegment, train_cached)
        val trainLabelsAndPredictions = (trainPartialPredictions zip currentPredictionsTrain).map {
          case ((point, segmentPrediction), currentPrediction) =>
            val prediction = sumType match {
              case "sum" => currentPrediction + segmentPrediction
              case "only" => segmentPrediction
              case "average" => (currentPrediction * (nElems - partialnElems) + segmentPrediction * partialnElems) / nElems

            }
          (point.label, prediction, point.data.id)
        }

        //trainPartialUnpersist()
        println("Saving Train Predictions")
//trainLabelsAndPredictions.sam
        if (trainLabelsAndPredictions.mapPartitionsWithIndex((i, p) => {
          println("save:")
          p.toSeq.groupBy(_._3).zipWithIndex.map{ case((id, d), j) =>
            NeuronUtils.saveLabelsAndPredictions(save_to + "/predictions/partial" + nElems + /*"/depth" + depth +*/ "/train/" + id,
              d.toIterator.map(x => (x._1, x._2)), dimensions_train(i)(j), s.toVerboseString, training_time
              /*,indexesAndGrads*/)
          }.toIterator
        }).count() != s.subvolumes.train.length) {
          println("Failed to save!")
          return
        }
        currentPredictionsTrain.unpersist()
        currentPredictionsTrain = trainLabelsAndPredictions.map(_._2).cache()
        currentPredictionsTrain.count()

        //val trainMSE = trainLabelsAndPredictions.map { case (v, p) => (v - p).sq}.mean() / 3
        //println("Train Mean Squared Error = " + trainMSE)
        println("Saved Train Predictions")

        // Test Error
        println("Predicting on test data")
        val /*(*/testPartialPredictions/*, testPartialUnpersist)*/ = MyTreeEnsembleModel.predictWithBroadcasts(mBroadcasts, partialSegment, test_cached)
        //val /*(*/testPartialPredictions/*, testPartialUnpersist)*/ = MyTreeEnsembleModel.predictSerial(partialSegment, test_cached)
        val testLabelsAndPredictions = (testPartialPredictions zip currentPredictionsTest).map {
          case ((point, segmentPrediction), currentPrediction) =>
            val prediction = if(partialSegment.isSum) currentPrediction + segmentPrediction
            else if(nElems == partialSegment.nElems) segmentPrediction
            else (currentPrediction * (nElems - partialSegment.nElems) + segmentPrediction * partialSegment.nElems) / nElems
            (point.label, prediction, point.data.id)
        }
        //testPartialPredictions.unpersist()
        println("Saving Test Predictions")
        if (testLabelsAndPredictions.mapPartitionsWithIndex((i, p) => {
          println("save:")
          p.toSeq.groupBy(_._3).zipWithIndex.map{ case((id, d), j) =>
            NeuronUtils.saveLabelsAndPredictions(save_to + "/predictions/partial" + nElems + /*"/depth" + depth +*/ "/test/" + id,
              d.toIterator.map(x => (x._1, x._2)), dimensions_test(i)(j), s.toVerboseString, training_time
              /*,indexesAndGrads*/)
          }.toIterator
        }).count() != s.subvolumes.test.length) {
          println("Failed to save!")
          return
        }

        currentPredictionsTest.unpersist()
        currentPredictionsTest = testLabelsAndPredictions.map(_._2).cache()
        currentPredictionsTest.count()

        //val testMSE = testLabelsAndPredictions.map { case (v, p) => (v - p).sq}.mean() / 3
        //println("Test Mean Squared Error = " + testMSE)
        println("Saved Test Predictions")

        mBroadcasts.foreach(_.unpersist())
      }
    //}


    println("Job complete. Saved to: " + save_to)
    println("Job took: " + ((System.currentTimeMillis() - start_time)/60000) + " minutes")

  }


  // -----------------------------------------------------------------------

  case class MalisSettings(learningRate:Double, subsampleProportion:Double, momentum:Double, treesPerIteration:Int)
  case class FeatureSettings(nBaseFeatures:Int, baseFeaturesAndOffsets:Seq[(Int, (Int, Int))], featureSubsetStrategy:String)
  case class Subvolumes(train: Seq[String], test:Seq[String])
  case class RunSettings(name:String, numExecutors:Int, maxMemoryInMB:Int, data_root:String, save_to:String, /*localDir: String,*/
                         subvolumes:Subvolumes, bagging:Double,
                         /*impurity:MyImpurity,*/ maxDepth:Int, maxBins:Int, initialModel:InitialModel,
                          master:String, save_model_to:String,
                         iterations:Int, saveGradients:Boolean, testPartialModels:Seq[Int], testDepths:Seq[Int],
                         useNodeIdCache:Boolean, malisSettings:MalisSettings, seed:Int, leafSize:Int, featureSettings:FeatureSettings) {
    //val nBaseFeatures = offsetMultiplier.length
    def toVerboseString =
      "RunSettings:\n" +
        " name = " + name + "\n" +
        " numExecutors = " + numExecutors + "\n" +
    " maxMemoryInMB = " + maxMemoryInMB + "\n" +
      /*" localDir = " + localDir + "\n" +*/
      " data_root = "     + data_root + "\n" +
      " save_to = "       + save_to + "\n" +
      " save_model_to = " + save_model_to + "\n" +
      " train_subvolumes = "    + subvolumes.train.toList + "\n" +
      " test_subvolumes = "    + subvolumes.test.toList + "\n" +
      " featureSubsetStrategy = "    + featureSettings.featureSubsetStrategy + "\n" +
      //" impurity = "    + impurity + "\n" +
      " maxDepth = "    + maxDepth + "\n" +
      " maxBins = "    + maxBins + "\n" +
      " nBaseFeatures = "    + featureSettings.nBaseFeatures + "\n" +
      " initialModel = "    + initialModel + "\n" +
      " treesPerIteration = "    + malisSettings.treesPerIteration + "\n" +
      " baseFeaturesAndOffsets: count = "  + featureSettings.baseFeaturesAndOffsets.length + "; values = {" + featureSettings.baseFeaturesAndOffsets.take(10).map(_.toString).reduce(_ + ", " + _) + ", ...}" + "\n" +
    " master = "    + master + "\n" +
      " learningRate = "    + malisSettings.learningRate + "\n" +
      " iterations = "   + iterations + "\n" +
      " saveGradients = " + saveGradients + "\n" +
      " testPartialModels = " + testPartialModels + "\n" +
      " testDepths = (IGNORED!) " + testDepths + "\n" +
      " useNodeIdCache = " + useNodeIdCache + "\n" +
      " bagging = " + bagging + "\n" +
      " subsampleProportion = " + malisSettings.subsampleProportion + "\n" +
      " momentum = " + malisSettings.momentum + "\n" +
      " seed = " + seed + "\n" +
      " leafSize = " + leafSize
  }

  def getSettingsFromArgs(args:Array[String]):RunSettings = {
//    println("Called with args:")
//    args.foreach(println)

    val m = args.map(_.split("=")).map(arr => arr(0) -> (if(arr.length>1) arr(1) else "")).toMap

    val train_subvolumes    = {
      val str = m.getOrElse("subvolumes_train", new File("/isbi_data").listFiles().map(_.getName).sorted.take(1).reduce(_ + "," + _))
      val idx = str.indexOf("*")
      if(idx != -1) Array.fill(str.substring(idx + 1).toInt)(str.substring(0, idx))
      else str.split(",")
    }
    val test_subvolumes    = {
      val str = m.getOrElse("subvolumes_test", new File("/isbi_data").listFiles().map(_.getName).sorted.drop(1).take(1).reduce(_ + "," + _))
      val idx = str.indexOf("*")
      if(idx != -1) Array.fill(str.substring(idx + 1).toInt)(str.substring(0, idx))
      else str.split(",")
    }
    def strToRange(s:String) = {
      if (s.take(1) == "{") {
        val inner = s.tail.init
        inner.split("|".head).map(_.toInt).toSeq
      } else {
        val split = s.split(":")
        split.length match {
          case 1 => Seq(s.toInt)
          case 2 => split(0).toInt until split(1).toInt
          case 3 => split(0).toInt until split(2).toInt by split(1).toInt
        }
      }
    }
    val defaultBaseFeaturesAndOffsets =
      "0:6({-32|-16|-8|-4|-2|0|2|4|8|16|32},{-32|-16|-8|-4|-2|0|2|4|8|16|32});6:12({-64|-32|-16|-8|-4|0|4|8|16|32|64},{-64|-32|-16|-8|-4|0|4|8|16|32|64});12:18({-64|-32|-16|-8|0|8|16|32|64},{-64|-32|-16|-8|0|8|16|32|64});18:24({-64|-32|-16|0|16|32|64},{-64|-32|-16|0|16|32|64});24:30({-64|-32|0|32|64},{-64|-32|0|32|64});30:36({-64|0|64},{-64|0|64})"

    RunSettings(
      name  = m.getOrElse("name", "expt"),
      numExecutors  = m.getOrElse("numExecutors", "1").toInt,
      maxMemoryInMB = m.getOrElse("maxMemoryInMB", "500").toInt,
      data_root     = m.getOrElse("data_root",     "/isbi_data"),
      save_to       = m.getOrElse("save_to",       "/isbi_predictions_local"),
      save_model_to = m.getOrElse("save_model_to", ""),
      //localDir      = m.getOrElse("localDir",     "/tmp"),
      //subvolumes    = m.getOrElse("subvolumes",    "000,001,010,011,100,101,110,111").split(",").toArray,
      subvolumes    = Subvolumes(train_subvolumes, test_subvolumes),
      //impurity      = MyImpurities.fromString(m.getOrElse("impurity", "variance")),
      maxDepth      = m.getOrElse("maxDepth",      "10").toInt,
      maxBins       = m.getOrElse("maxBins",       "10").toInt,

      initialModel  = //InitialLoadedModel("/masters_models/2015-04-16 17-42-35"),
                      if(m.contains("loadModel")) InitialLoadedModel(m("loadModel"))
                      else InitialTrainModel(m.getOrElse("initialTrees",  "1").toInt),
      //dimOffsets    = m.getOrElse("dimOffsets",    "0").split(",").map(_.toInt),
      //offsetMultiplier    = m.getOrElse("offsetMultiplier",    "1,1,1,1,1,1,2,2,2,2,2,2,4,4,4,4,4,4,8,8,8,8,8,8").split(",").map(_.toInt),

      featureSettings = FeatureSettings(
        nBaseFeatures = m.getOrElse("nBaseFeatures", "36").toInt,
        baseFeaturesAndOffsets = m.getOrElse("features", defaultBaseFeaturesAndOffsets).split(";").flatMap{ s =>
          val entry = s.splitAt(s.indexOf("("))
          val features = strToRange(entry._1)
          val offsets = entry._2.split("(".head).tail.flatMap{ o =>
            val lr = o.init.split(",")
            for(x <- strToRange(lr(0)); y <- strToRange(lr(1))) yield (x,y)
          }

          for(f <- features; o <- offsets) yield (f, o)
        },
        featureSubsetStrategy = m.getOrElse("featureSubsetStrategy", "all")
      ),
      master        = m.getOrElse("master",        "local"), // use empty string to not setdata_
      iterations    = m.getOrElse("iterations", "0").toInt,
      saveGradients = m.getOrElse("saveGradients", "false").toBoolean,
      testPartialModels = {
        val in = m.getOrElse("testPartialModels", "")
        if (in=="none") Seq(-1)
        else in.split(",").filter(! _.isEmpty).map(_.toInt)
      },
      testDepths    = m.getOrElse("testDepths", "").split(",").filter(! _.isEmpty).map(_.toInt),
      useNodeIdCache = m.getOrElse("useNodeIdCache", "true").toBoolean,
      bagging = m.getOrElse("bagging", "0.5").toDouble,
      malisSettings = MalisSettings(
        treesPerIteration = m.getOrElse("treesPerIteration", "10").toInt,
        learningRate     = m.getOrElse("learningRate",     "1").toDouble,
        subsampleProportion = m.getOrElse("subsampleProportion", "1").toDouble,
        momentum = m.getOrElse("momentum", "0").toDouble
      ),
      seed = m.getOrElse("seed", "1").toInt,
      leafSize = m.getOrElse("leafSize", "1").toInt

    )
  }
}