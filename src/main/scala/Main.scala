import java.io
import java.text.SimpleDateFormat
import java.util.{Date, Calendar}

import main.scala.org.apache.spark.mllib.tree.model.MyModel
import org.apache.spark.mllib.tree.model._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.loss.MalisLoss
import org.apache.spark.mllib.tree.{RandomForest, MyGradientBoostedTrees, MyRandomForest, NeuronUtils}
import org.apache.spark.mllib.tree.configuration.{MyBoostingStrategy, MyStrategy, Strategy}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.tree.impurity._
import org.apache.spark.mllib.tree.configuration.QuantileStrategy._
import org.apache.spark.mllib.tree.configuration.Algo._

object Main {

  def main(args:Array[String]) {
    val start_time = System.currentTimeMillis()

    //------------------------ Init ---------------------------------------
    val s = getSettingsFromArgs(args)
    println("Settings:\n" + s.toVerboseString)

    val offsets = for (x <- s.dimOffsets; y <- s.dimOffsets; z <- s.dimOffsets) yield (x, y, z)
    val nFeatures = s.nBaseFeatures * offsets.length
    val conf = new SparkConf().setAppName("Hello").set("spark.shuffle.spill", "false").set("spark.local.dir", s.localDir)
    if (!s.master.isEmpty) conf.setMaster(s.master)
    val sc = new SparkContext(conf)

    val timestr = new SimpleDateFormat("yyyy-MM-dd HH-mm-ss").format(new Date())
    val save_to = s.save_to + "/" + timestr

//    //-------------------------- Train -------------------------------------
    val (splits, bins) = NeuronUtils.getSplitsAndBins(s.subvolumes, s.nBaseFeatures, s.data_root, s.maxBins, offsets)
    val (train, dimensions_train) = NeuronUtils.loadData(sc, s.subvolumes, s.nBaseFeatures, s.data_root, s.maxBins, offsets, s.trainFraction, bins, fromFront = true)
    //train.persist(StorageLevel.MEMORY_ONLY_SER)
    val strategy = new MyStrategy(Regression, s.impurity, s.maxDepth, 2, s.maxBins, Sort, Map[Int, Int](), maxMemoryInMB = s.maxMemoryInMB, useNodeIdCache = s.useNodeIdCache)

    val model: MyEnsembleModel[_] = if (s.iterations == 0) {
      println("Training a Random Forest Model (no gradient boosting)")
      //    Random Forest
      MyRandomForest.trainRegressorFromTreePoints(train, strategy, s.initialTrees, s.featureSubsetStrategy: String, 1,
        nFeatures, dimensions_train.map(_.n_targets).sum, splits, bins)
    } else {
      println("Training a Gradient boosted model")
      //    Gradient Boosting
      val boostingStrategy = new MyBoostingStrategy(strategy, MalisLoss, s.initialTrees, s.iterations, s.treesPerIteration, s.malisGrad)
      //    val (model, grads, seg) = new MyGradientBoostedTrees(boostingStrategy).run(train, boostingStrategy, nFeatures,
      //      dimensions_train.map(_.n_targets).sum, splits, bins, s.featureSubsetStrategy)
      new MyGradientBoostedTrees(boostingStrategy).run(train, boostingStrategy, nFeatures,
        dimensions_train.map(_.n_targets).sum, splits, bins, s.subsampleProportion, s.featureSubsetStrategy, if(s.saveGradients) save_to + "/gradients" else null)

    } /*else {
      println(s.mode + " is not a valid mode!")
      ???
    }*/
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


    //-------------------------- Test ---------------------------------------
    val (test, dimensions_test) = NeuronUtils.loadData(sc, s.subvolumes, s.nBaseFeatures, s.data_root, s.maxBins, offsets, 1 - s.trainFraction, bins, fromFront = false)

    val allPartialModels:Seq[MyEnsembleModel[_]] = model.getPartialModels
    println("There are " + allPartialModels.size + " partial models")

    val (train_cached, _) = NeuronUtils.cached(train)
    val (test_cached, _) = NeuronUtils.cached(test)

    val testPartialModels = (s.testPartialModels :+ allPartialModels.size).distinct //ensure that the final model is tested
    val testDepths = if(s.testDepths.isEmpty) Seq(Integer.MAX_VALUE) else s.testDepths

    var i=0
    while(i < testPartialModels.size) { //shit goes weird when I use a for loop. WHY? todo:INVESTIGATE
      val partialModel = allPartialModels(testPartialModels(i)-1)
      i+=1
      var j = 0
      while (j < testDepths.size) { // WHYYYY???
        val depth = testDepths(j)
        j += 1

        val nElems = partialModel.nElems
        partialModel.elems.head match {
          case _: MyDecisionTreeModel => partialModel.elems.foreach(_.asInstanceOf[MyDecisionTreeModel].capDepth(depth))
          case _: MyRandomForestModel => partialModel.elems.foreach(_.asInstanceOf[MyRandomForestModel].trees.foreach(_.capDepth(depth)))
        }
        println("\nTesting partial model with " + nElems + " elements, at depth " + depth)

        // Training Error
        val trainLabelsAndPredictions = train_cached.map { point =>
          val features = Array.tabulate[Double](nFeatures)(f => point.features(f))
          val prediction = partialModel.predict(Vectors.dense(features))
          (point.label, prediction)
        }.cache()
        if (trainLabelsAndPredictions.mapPartitionsWithIndex((i, p) => {
          println("save:")
          NeuronUtils.saveLabelsAndPredictions(save_to + "/predictions/partial" + nElems + "/depth" + depth + "/train/" + i, p, dimensions_train(i), s.toVerboseString, training_time
            /*,indexesAndGrads*/)
          Iterator("foo")
        }).count() != s.subvolumes.length) {
          println("Failed to save!")
          return
        }
        val trainMSE = trainLabelsAndPredictions.map { case (v, p) => (v - p).sq}.mean() / 3
        println("Train Mean Squared Error = " + trainMSE)
        trainLabelsAndPredictions.unpersist()


        // Test Error
        val testLabelsAndPredictions = test_cached.map { point =>
          val features = Array.tabulate[Double](nFeatures)(f => point.features(f))
          val prediction = partialModel.predict(Vectors.dense(features))
          (point.label, prediction)
        }.cache()
        if (testLabelsAndPredictions.mapPartitionsWithIndex((i, p) => {
          println("save:")
          NeuronUtils.saveLabelsAndPredictions(save_to + "/predictions/partial" + nElems + "/depth" + depth + "/test/" + i, p, dimensions_test(i), s.toVerboseString, training_time
            /*,indexesAndGrads*/)
          Iterator("foo")
        }).count() != s.subvolumes.length) {
          println("Failed to save!")
          return
        }
        val testMSE = testLabelsAndPredictions.map { case (v, p) => (v - p).sq}.mean() / 3
        println("Test Mean Squared Error = " + testMSE)
        testLabelsAndPredictions.unpersist()
      }
    }

    println("Job complete. Saved to: " + s.save_to + "/" + timestr)

    println("Job took: " + ((System.currentTimeMillis() - start_time)/60000) + " minutes")
  }


  // -----------------------------------------------------------------------

  case class RunSettings(maxMemoryInMB:Int, data_root:String, save_to:String, localDir: String, subvolumes:Seq[String], featureSubsetStrategy:String,
                         impurity:MyImpurity, maxDepth:Int, maxBins:Int, nBaseFeatures:Int, initialTrees:Int, treesPerIteration:Int,
                         dimOffsets:Seq[Int], master:String, trainFraction:Double, malisGrad:Double,
                         iterations:Int, saveGradients:Boolean, testPartialModels:Seq[Int], testDepths:Seq[Int], useNodeIdCache:Boolean, subsampleProportion:Double) {
    def toVerboseString =
      "RunSettings:\n" +
      " maxMemoryInMB = " + maxMemoryInMB + "\n" +
      " localDir = " + localDir + "\n" +
      " data_root = "     + data_root + "\n" +
      " save_to = "       + save_to + "\n" +
      " subvolumes = "    + subvolumes.toList + "\n" +
      " featureSubsetStrategy = "    + featureSubsetStrategy + "\n" +
      " impurity = "    + impurity + "\n" +
      " maxDepth = "    + maxDepth + "\n" +
      " maxBins = "    + maxBins + "\n" +
      " nBaseFeatures = "    + nBaseFeatures + "\n" +
    "   initialTrees = "    + initialTrees + "\n" +
      " treesPerIteration = "    + treesPerIteration + "\n" +
      " dimOffsets = "    + dimOffsets.toList + "\n" +
      " master = "    + master + "\n" +
      " trainFraction = "    + trainFraction + "\n" +
      " malisGrad = "    + malisGrad + "\n" +
      " iterations = "   + iterations + "\n" +
      " saveGradients = " + saveGradients + "\n" +
      " testPartialModels = " + testPartialModels + "\n" +
      " testDepths = " + testDepths + "\n" +
      " useNodeIdCache = " + useNodeIdCache + "\n" +
    " subsampleProportion = " + subsampleProportion
  }

  def getSettingsFromArgs(args:Array[String]):RunSettings = {
//    println("Called with args:")
//    args.foreach(println)

    val m = args.map(_.split("=")).map(arr => arr(0) -> (if(arr.length>1) arr(1) else "")).toMap
    RunSettings(
      maxMemoryInMB = m.getOrElse("maxMemoryInMB", "500").toInt,
      data_root     = m.getOrElse("data_root",     "/masters_data/spark/im1/split_2"),
      save_to       = m.getOrElse("save_to",       "/masters_predictions"),
      localDir      = m.getOrElse("localDir",     "/tmp"),
      //subvolumes    = m.getOrElse("subvolumes",    "000,001,010,011,100,101,110,111").split(",").toArray,
      subvolumes    = {
        val str = m.getOrElse("subvolumes",    "011")
        val idx = str.indexOf("*")
        if(idx != -1) Array.fill(str.substring(idx + 1).toInt)(str.substring(0, idx))
        else str.split(",")
      },
      featureSubsetStrategy = m.getOrElse("featureSubsetStrategy", "sqrt"),
      impurity      = MyImpurities.fromString(m.getOrElse("impurity", "variance")),
      maxDepth      = m.getOrElse("maxDepth",      "14").toInt,
      maxBins       = m.getOrElse("maxBins",       "100").toInt,
      nBaseFeatures = m.getOrElse("nBaseFeatures", "30").toInt,
      initialTrees  = m.getOrElse("initialTrees",          "10").toInt,
      treesPerIteration = m.getOrElse("treesPerIteration", "10").toInt,
      dimOffsets    = m.getOrElse("dimOffsets",    "0").split(",").map(_.toInt),
      master        = m.getOrElse("master",        "local"), // use empty string to not setdata_
      trainFraction = m.getOrElse("trainFraction", "0.5").toDouble,
      malisGrad     = m.getOrElse("malisGrad",     "100").toDouble,
      iterations    = m.getOrElse("iterations", "1").toInt,
      saveGradients = m.getOrElse("saveGradients", "false").toBoolean,
      testPartialModels = m.getOrElse("testPartialModels", "").split(",").map(_.toInt),
      testDepths    = m.getOrElse("testDepths", "").split(",").map(_.toInt),
      useNodeIdCache = m.getOrElse("useNodeIdCache", "true").toBoolean,
      subsampleProportion = m.getOrElse("subsampleProportion", "1").toDouble
    )
  }
}