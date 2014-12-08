import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.{MyRandomForest, NeuronUtils}
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.tree.impurity.{Gini, Entropy, Impurity}
import org.apache.spark.mllib.tree.configuration.QuantileStrategy._
import org.apache.spark.mllib.tree.configuration.Algo._

object Main {

  def main(args:Array[String]) {
    //------------------------ Init ---------------------------------------

    val s = getSettingsFromArgs(args)
    println("Settings:\n" + s)

    val offsets = for (x <- s.dimOffsets; y <- s.dimOffsets; z <- s.dimOffsets) yield (x, y, z)
    val nFeatures = s.nBaseFeatures * offsets.length
    val conf = new SparkConf().setAppName("Hello")
    if (!s.master.isEmpty) conf.setMaster(s.master)
    val sc = new SparkContext(conf)


    //-------------------------- Train -------------------------------------
    val (splits, bins) = NeuronUtils.getSplitsAndBins(s.subvolumes, s.nBaseFeatures, s.data_root, s.maxBins, offsets)
    val train = NeuronUtils.loadData(sc, s.subvolumes, s.nBaseFeatures, s.data_root, s.maxBins, offsets, 0.2, bins, fromFront = true)
    //train.persist(StorageLevel.MEMORY_ONLY_SER)
    val strategy = new Strategy(Classification, s.impurity, s.maxDepth, 2, s.maxBins, Sort, Map[Int, Int](), maxMemoryInMB = s.maxMemoryInMB)
    val model = MyRandomForest.trainClassifierFromTreePoints(train, strategy, s.nTrees, s.featureSubsetStrategy: String, 1,
      nFeatures, 400000, splits, bins)

    println("trained.")

    //-------------------------- Test ---------------------------------------
    val test = NeuronUtils.loadData(sc, s.subvolumes, s.nBaseFeatures, s.data_root, s.maxBins, offsets, 0.8, bins, fromFront = false)
    val labelsAndPredictions = test.map { point =>
      val features = Array.tabulate[Double](nFeatures)(f => point.features(f))
      val prediction = model.predict(Vectors.dense(features))
      (point.label, prediction)
    }
    val testMSE = labelsAndPredictions.map { case (v, p) => math.pow(v - p, 2)}.mean()
    println("Test Mean Squared Error = " + testMSE)
    println("Learned regression tree model:\n" + model)
  }





  // -----------------------------------------------------------------------

  case class RunSettings(maxMemoryInMB:Int, data_root:String, subvolumes:Array[String], featureSubsetStrategy:String,
                         impurity:Impurity, maxDepth:Int, maxBins:Int, nBaseFeatures:Int, nTrees:Int,
                         dimOffsets:Array[Int], master:String)

  def getSettingsFromArgs(args:Array[String]):RunSettings = {
    println("Called with args:")
    args.foreach(println)

    val m = args.map(_.split("=")).map(arr => arr(0) -> (if(arr.length>1) arr(1) else "")).toMap
    val impurityMap = Seq("entropy" -> Entropy, "gini" -> Gini).toMap
    RunSettings(
      maxMemoryInMB = m.getOrElse("maxMemoryInMB", "1000").toInt,
      data_root     = m.getOrElse("data_root",     "/masters_data/spark/im1/split_2"),
      //subvolumes    = m.getOrElse("subvolumes",    "000,001,010,011,100,101,110,111").split(",").toArray,
      subvolumes    = m.getOrElse("subvolumes",    "000").split(",").toArray,
      featureSubsetStrategy = m.getOrElse("featureSubsetStrategy", "sqrt"),
      impurity      = impurityMap(m.getOrElse("impurity", "entropy")),
      maxDepth      = m.getOrElse("maxDepth",      "14").toInt,
      maxBins       = m.getOrElse("maxBins",       "100").toInt,
      nBaseFeatures = m.getOrElse("nBaseFeatures", "30").toInt,
      nTrees        = m.getOrElse("nTrees",        "50").toInt,
      dimOffsets    = m.getOrElse("dimOffsets",    "0").split(",").map(_.toInt).toArray,
      master        = m.getOrElse("master",        "local") // use empty string to not setdata_
    )
  }
}