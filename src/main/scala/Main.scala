import java.io
import java.text.SimpleDateFormat
import java.util.{Date, Calendar}

import main.scala.org.apache.spark.mllib.tree.model.MyModel
import org.apache.spark.mllib.tree.model.MyEnsembleModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.loss.MalisLoss
import org.apache.spark.mllib.tree.{MyGradientBoostedTrees, MyRandomForest, NeuronUtils}
import org.apache.spark.mllib.tree.configuration.{MyBoostingStrategy, MyStrategy, Strategy}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.tree.impurity.{Gini, Entropy, MyImpurity, MyImpurities}
import org.apache.spark.mllib.tree.configuration.QuantileStrategy._
import org.apache.spark.mllib.tree.configuration.Algo._

object Main {

  def main(args:Array[String]) {
    //------------------------ Init ---------------------------------------

    val s = getSettingsFromArgs(args)
    println("Settings:\n" + s)

    val offsets = for (x <- s.dimOffsets; y <- s.dimOffsets; z <- s.dimOffsets) yield (x, y, z)
    val nFeatures = s.nBaseFeatures * offsets.length
    val conf = new SparkConf().setAppName("Hello").set("spark.shuffle.spill", "false")
    if (!s.master.isEmpty) conf.setMaster(s.master)
    val sc = new SparkContext(conf)



    //-------------------------- Train -------------------------------------
    val (splits, bins) = NeuronUtils.getSplitsAndBins(s.subvolumes, s.nBaseFeatures, s.data_root, s.maxBins, offsets)
    val (train, dimensions_train) = NeuronUtils.loadDataCached(sc, s.subvolumes, s.nBaseFeatures, s.data_root, s.maxBins, offsets, s.trainFraction, bins, fromFront = true)
    //train.persist(StorageLevel.MEMORY_ONLY_SER)
    val strategy = new MyStrategy(Regression, s.impurity, s.maxDepth, 2, s.maxBins, Sort, Map[Int, Int](), maxMemoryInMB = s.maxMemoryInMB)

    val model: MyEnsembleModel[_] = if (s.mode == "RandomForest") {
      //    Random Forest
      MyRandomForest.trainRegressorFromTreePoints(train, strategy, s.nTrees, s.featureSubsetStrategy: String, 1,
        nFeatures, dimensions_train.map(_.n_targets).sum, splits, bins)
    } else if (s.mode == "MALIS") {
      //    Gradient Boosting
      val boostingStrategy = new MyBoostingStrategy(strategy, MalisLoss, 5, 10, math.pow(20, -6) * 2)
      //    val (model, grads, seg) = new MyGradientBoostedTrees(boostingStrategy).run(train, boostingStrategy, nFeatures,
      //      dimensions_train.map(_.n_targets).sum, splits, bins, s.featureSubsetStrategy)
      new MyGradientBoostedTrees(boostingStrategy).run(train, boostingStrategy, nFeatures,
        dimensions_train.map(_.n_targets).sum, splits, bins, s.featureSubsetStrategy)

    } else {
      println(s.mode + " is not a valid mode!")
      ???
    }
    println("trained.")



    //-------------------------- Test ---------------------------------------
    val (test, dimensions_test) = NeuronUtils.loadDataCached(sc, s.subvolumes, s.nBaseFeatures, s.data_root, s.maxBins, offsets, 1 - s.trainFraction, bins, fromFront = false)
    val timestr = new SimpleDateFormat("yyyy-MM-dd HH-mm-ss").format(new Date())

    val allPartialModels:Seq[MyEnsembleModel[_]] = model.getPartialModels

    var partialIndex = 0
    while(partialIndex < allPartialModels.size) { // If I use a for loop then the compiler does some optimization and all hell breaks loos
      val partialModel = allPartialModels(partialIndex)
      //for(/*partialModel <- partialModels*/ _ <- 0 until 1) {
      val nElems = partialModel.nElems
      println("\nTesting partial model with " + nElems + " elements")

      // Training Error
      val trainLabelsAndPredictions = train.map { point =>
        val features = Array.tabulate[Double](nFeatures)(f => point.features(f))
        val prediction = partialModel.predict(Vectors.dense(features))
        (point.label, prediction)
      }.cache()
      trainLabelsAndPredictions.mapPartitionsWithIndex((i, p) => {
        println("save:")
        NeuronUtils.saveLabelsAndPredictions(s.save_to + "/" + timestr + "/partial" + nElems + "/train/" + i, p, dimensions_train(i), s.toString
          /*,indexesAndGrads*/)
        Iterator("foo")
      }).count()
      val trainMSE = trainLabelsAndPredictions.map { case (v, p) => (v - p).sq}.mean() / 3
      println("Train Mean Squared Error = " + trainMSE)
      trainLabelsAndPredictions.unpersist()

      // Test Error
      val testLabelsAndPredictions = test.map { point =>
        val features = Array.tabulate[Double](nFeatures)(f => point.features(f))
        val prediction = partialModel.predict(Vectors.dense(features))
        (point.label, prediction)
      }.cache()
      testLabelsAndPredictions.mapPartitionsWithIndex((i, p) => {
        println("save:")
        NeuronUtils.saveLabelsAndPredictions(s.save_to + "/" + timestr + "/partial" + nElems + "/test/" + i, p, dimensions_test(i), s.toString
          /*,indexesAndGrads*/)
        Iterator("foo")
      }).count()
      val testMSE = testLabelsAndPredictions.map { case (v, p) => (v - p).sq}.mean() / 3
      println("Test Mean Squared Error = " + testMSE)
      testLabelsAndPredictions.unpersist()


      // If ensemble is only size 5 include all partials. Otherwise include every fifth partial
      if(model.nElems <= 5) partialIndex += 1
      else if(partialIndex < allPartialModels.size - 6) partialIndex += 5
      else if(partialIndex != allPartialModels.size-1) partialIndex = allPartialModels.size-1
      else partialIndex = allPartialModels.size //break out of loop
    }

  }


  // -----------------------------------------------------------------------

  case class RunSettings(maxMemoryInMB:Int, data_root:String, save_to:String, subvolumes:Seq[String], featureSubsetStrategy:String,
                         impurity:MyImpurity, maxDepth:Int, maxBins:Int, nBaseFeatures:Int, nTrees:Int,
                         dimOffsets:Seq[Int], master:String, trainFraction:Double, mode:String)

  def getSettingsFromArgs(args:Array[String]):RunSettings = {
    println("Called with args:")
    args.foreach(println)

    val m = args.map(_.split("=")).map(arr => arr(0) -> (if(arr.length>1) arr(1) else "")).toMap
    RunSettings(
      maxMemoryInMB = m.getOrElse("maxMemoryInMB", "500").toInt,

      data_root     = m.getOrElse("data_root",     "/masters_data/spark/im1/split_2"),
      save_to       = m.getOrElse("save_to",       "/masters_predictions"),
      //subvolumes    = m.getOrElse("subvolumes",    "000,001,010,011,100,101,110,111").split(",").toArray,
      subvolumes    = m.getOrElse("subvolumes",    "000").split(","),
      featureSubsetStrategy = m.getOrElse("featureSubsetStrategy", "sqrt"),
      impurity      = MyImpurities.fromString(m.getOrElse("impurity", "variance")),
      maxDepth      = m.getOrElse("maxDepth",      "14").toInt,
      maxBins       = m.getOrElse("maxBins",       "100").toInt,
      nBaseFeatures = m.getOrElse("nBaseFeatures", "30").toInt,
      nTrees        = m.getOrElse("nTrees",        "50").toInt,
      dimOffsets    = m.getOrElse("dimOffsets",    "0").split(",").map(_.toInt),
      master        = m.getOrElse("master",        "local"), // use empty string to not setdata_
      trainFraction = m.getOrElse("trainFraction", "0.1").toDouble,
      mode          = m.getOrElse("mode", "RandomForest")
    )
  }
}