import java.io
import java.text.SimpleDateFormat
import java.util.{Date, Calendar}

import main.scala.org.apache.spark.mllib.tree.model.MyModel
import org.apache.spark.mllib.tree.model.{MyGradientBoostedTreesModel, MyRandomForestModel, MyEnsembleModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.loss.MalisLoss
import org.apache.spark.mllib.tree.{MyGradientBoostedTrees, MyRandomForest, NeuronUtils}
import org.apache.spark.mllib.tree.configuration.{MyBoostingStrategy, MyStrategy, Strategy}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.tree.impurity.{Gini, Entropy, MyImpurity, MyImpurities}
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
    val conf = new SparkConf().setAppName("Hello").set("spark.shuffle.spill", "false")
    if (!s.master.isEmpty) conf.setMaster(s.master)
    val sc = new SparkContext(conf)

    val timestr = new SimpleDateFormat("yyyy-MM-dd HH-mm-ss").format(new Date())
    val save_to = s.save_to + "/" + timestr

    //-------------------------- Train -------------------------------------
    val (splits, bins) = NeuronUtils.getSplitsAndBins(s.subvolumes, s.nBaseFeatures, s.data_root, s.maxBins, offsets)
    val (train, dimensions_train) = NeuronUtils.loadData(sc, s.subvolumes, s.nBaseFeatures, s.data_root, s.maxBins, offsets, s.trainFraction, bins, fromFront = true)
    //train.persist(StorageLevel.MEMORY_ONLY_SER)
    val strategy = new MyStrategy(Regression, s.impurity, s.maxDepth, 2, s.maxBins, Sort, Map[Int, Int](), maxMemoryInMB = s.maxMemoryInMB)

    val model: MyEnsembleModel[_] = if (s.iterations == 1) {
      //    Random Forest
      MyRandomForest.trainRegressorFromTreePoints(train, strategy, s.nTrees, s.featureSubsetStrategy: String, 1,
        nFeatures, dimensions_train.map(_.n_targets).sum, splits, bins)
    } else {
      //    Gradient Boosting
      val boostingStrategy = new MyBoostingStrategy(strategy, MalisLoss, s.iterations, s.nTrees, s.malisGrad)
      //    val (model, grads, seg) = new MyGradientBoostedTrees(boostingStrategy).run(train, boostingStrategy, nFeatures,
      //      dimensions_train.map(_.n_targets).sum, splits, bins, s.featureSubsetStrategy)
      new MyGradientBoostedTrees(boostingStrategy).run(train, boostingStrategy, nFeatures,
        dimensions_train.map(_.n_targets).sum, splits, bins, s.featureSubsetStrategy, if(s.saveGradients) save_to + "/gradients" else null)

    } /*else {
      println(s.mode + " is not a valid mode!")
      ???
    }*/
    println("trained.")



    //-------------------------- Test ---------------------------------------
    val (test, dimensions_test) = NeuronUtils.loadData(sc, s.subvolumes, s.nBaseFeatures, s.data_root, s.maxBins, offsets, 1 - s.trainFraction, bins, fromFront = false)

    val allPartialModels:Seq[MyEnsembleModel[_]] = model.getPartialModels

    val (train_cached, _) = NeuronUtils.cached(train)
    val (test_cached, _) = NeuronUtils.cached(test)

    val testPartialModels = s.testPartialModels + (allPartialModels.size-1) //ensure that the final model is tested
    val testDepths = if(s.testDepths.isEmpty) Set(Integer.MAX_VALUE) else s.testDepths

    for(i <- testPartialModels; depth <- testDepths) {
      val partialModel = allPartialModels(i)
      val nElems = partialModel.nElems
      partialModel match {
        case m:MyRandomForestModel => m.trees.foreach(_.capDepth(depth))
        case m:MyGradientBoostedTreesModel => m.elems.foreach(_.trees.foreach(_.capDepth(depth)))
      }
      println("\nTesting partial model with " + nElems + " elements, at depth " + depth)

      // Training Error
      val trainLabelsAndPredictions = train_cached.map { point =>
        val features = Array.tabulate[Double](nFeatures)(f => point.features(f))
        val prediction = partialModel.predict(Vectors.dense(features))
        (point.label, prediction)
      }.cache()
      println("Saving...")
      if(trainLabelsAndPredictions.mapPartitionsWithIndex((i, p) => {
        println("save:")
        NeuronUtils.saveLabelsAndPredictions(save_to + "/predictions/partial" + nElems + "/train/" + i, p, dimensions_train(i), s.toVerboseString
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
      println("Saving...")
      if(testLabelsAndPredictions.mapPartitionsWithIndex((i, p) => {
        println("save:")
        NeuronUtils.saveLabelsAndPredictions(save_to + "/predictions/partial" + nElems + "/test/" + i, p, dimensions_test(i), s.toVerboseString
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



    println("Job complete. Saved to: " + s.save_to + "/" + timestr)
    println("Job took: " + ((System.currentTimeMillis() - start_time)/60000) + " minutes")
  }


  // -----------------------------------------------------------------------

  case class RunSettings(maxMemoryInMB:Int, data_root:String, save_to:String, subvolumes:Seq[String], featureSubsetStrategy:String,
                         impurity:MyImpurity, maxDepth:Int, maxBins:Int, nBaseFeatures:Int, nTrees:Int,
                         dimOffsets:Seq[Int], master:String, trainFraction:Double, malisGrad:Double,
                         iterations:Int, saveGradients:Boolean, testPartialModels:Set[Int], testDepths:Set[Int]) {
    def toVerboseString =
      "RunSettings:\n" +
      " maxMemoryInMB = " + maxMemoryInMB + "\n" +
      " data_root = "     + data_root + "\n" +
      " save_to = "       + save_to + "\n" +
      " subvolumes = "    + subvolumes.toList + "\n" +
      " featureSubsetStrategy = "    + featureSubsetStrategy + "\n" +
      " impurity = "    + impurity + "\n" +
      " maxDepth = "    + maxDepth + "\n" +
      " maxBins = "    + maxBins + "\n" +
      " nBaseFeatures = "    + nBaseFeatures + "\n" +
      " nTrees = "    + nTrees + "\n" +
      " dimOffsets = "    + dimOffsets.toList + "\n" +
      " master = "    + master + "\n" +
      " trainFraction = "    + trainFraction + "\n" +
      " malisGrad = "    + malisGrad + "\n" +
      " iterations = "   + iterations + "\n" +
      " saveGradients = " + saveGradients + "\n" +
      " testPartialModels = " + testPartialModels +
      " testDepths = " + testDepths
  }

  def getSettingsFromArgs(args:Array[String]):RunSettings = {
//    println("Called with args:")
//    args.foreach(println)

    val m = args.map(_.split("=")).map(arr => arr(0) -> (if(arr.length>1) arr(1) else "")).toMap
    RunSettings(
      maxMemoryInMB = m.getOrElse("maxMemoryInMB", "500").toInt,

      data_root     = m.getOrElse("data_root",     "/masters_data/spark/im1/split_2"),
      save_to       = m.getOrElse("save_to",       "/masters_predictions"),
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
      nTrees        = m.getOrElse("nTrees",        "10").toInt,
      dimOffsets    = m.getOrElse("dimOffsets",    "0").split(",").map(_.toInt),
      master        = m.getOrElse("master",        "local"), // use empty string to not setdata_
      trainFraction = m.getOrElse("trainFraction", "0.5").toDouble,
      malisGrad     = m.getOrElse("malisGrad",     "1").toDouble,
      iterations    = m.getOrElse("iterations", "2").toInt,
      saveGradients = m.getOrElse("saveGradients", "false").toBoolean,
      testPartialModels = m.getOrElse("testPartialModels", "").split(",").map(_.toInt).toSet,
      testDepths    = m.getOrElse("testDepths", "").split(",").map(_.toInt).toSet
    )
  }
}