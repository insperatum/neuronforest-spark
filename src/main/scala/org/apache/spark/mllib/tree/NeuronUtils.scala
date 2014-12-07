package org.apache.spark.mllib.tree

import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.configuration.QuantileStrategy._
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impl.{MyDecisionTreeMetadata, BinnedFeatureData, MyTreePoint, RawFeatureData}
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.mllib.tree.model.{Bin, Split}

import org.apache.spark.rdd.RDD

import scala.io.Source

object NeuronUtils {
  def loadData(sc: SparkContext, subvolumes: Array[String], nBaseFeatures: Int, data_root: String,
               maxBins:Int, offsets:Array[(Int, Int, Int)], proportion: Double, fromFront: Boolean) = {
    //todo: use numFiles
    val rawFeaturesData = sc.parallelize(1 to subvolumes.size, subvolumes.size).mapPartitionsWithIndex((i, _) => {
      val features_file = data_root + "/" + subvolumes(i) + "/features.raw"
      Seq(new RawFeatureData(features_file, nBaseFeatures)).toIterator
    })
    rawFeaturesData.cache()

    val baseFeaturesRDD = rawFeaturesData.mapPartitions(_.next().toVectors)
    println("getting splits and bins")
    val (splits, bins) = getSplitsAndBins(baseFeaturesRDD, maxBins, nBaseFeatures, offsets.length)
    println(" found bins!")

    val data = rawFeaturesData.mapPartitionsWithIndex((i, f) => {
      val startTime = System.currentTimeMillis()

      val targets_file = data_root + "/" + subvolumes(i) + "/targets.txt"
      val n_targets_total = Source.fromFile(targets_file).getLines().size //todo: store this at the top of the file (OR GET FROM DIMENSIONS!)
      val n_targets = (n_targets_total * proportion).toInt
      val target_index_offset = if (fromFront) 0 else n_targets_total - n_targets

      val allTargets = Source.fromFile(targets_file).getLines().map(_.split(" ").map(_.toDouble))
      val targets = if (fromFront)
        allTargets.take(n_targets)
      else
        allTargets.drop(target_index_offset)

      val dimensions_file = data_root + "/" + subvolumes(i) + "/dimensions.txt"
      val dimensions = Source.fromFile(dimensions_file).getLines().map(_.split(" ").map(_.toInt)).toArray

      val size = dimensions(0)
      val step = (size(1) * size(2), size(2), 1)
      val min_idx = (dimensions(1)(0), dimensions(1)(1), dimensions(1)(2))
      val max_idx = (dimensions(2)(0), dimensions(2)(1), dimensions(2)(2))

      println("Targets from " + min_idx + " to " + max_idx)


      val seg_size = (max_idx._1 - min_idx._1 + 1, max_idx._2 - min_idx._2 + 1, max_idx._3 - min_idx._3 + 1)
      val seg_step = (seg_size._2 * seg_size._3, seg_size._3, 1)

      val binnedFeatureData = new BinnedFeatureData(f.next(), bins, seg_step, offsets)
      val d = targets.zipWithIndex.map { case (ts, i) =>
        val t = i + target_index_offset
        val y = ts(0)
        val example_idx =
          step._1 * (min_idx._1 + t / seg_step._1) +
            step._2 * (min_idx._2 + (t % seg_step._1) / seg_step._2) +
            (min_idx._3 + t % seg_step._2)
        //LabeledPoint(y, Vectors.dense(binnedFeatureData.arr.slice(example_idx * nFeatures, (example_idx + 1) * nFeatures).map(_.toDouble)))
        new MyTreePoint(y, null, binnedFeatureData, example_idx)
      }

      println("creating partition data took " + (System.currentTimeMillis() - startTime) + " ms")
      d
    })
    rawFeaturesData.unpersist()
    (data, splits, bins)
  }





  def getSplitsAndBins(featuress:RDD[org.apache.spark.mllib.linalg.Vector], maxBins:Int, nBaseFeatures:Int, nOffsets:Int):
  (Array[Array[Split]], Array[Array[Bin]]) = {
    println("getSplitsAndBins")
    val strategy = new Strategy(Classification, Gini, 0, 0, maxBins, Sort, Map[Int, Int]())
    val fakemetadata = MyDecisionTreeMetadata.buildMetadataFromFeatures(featuress, strategy, 50, "sqrt")
    val (rawFeatureSplits, rawFeatureBins) = MyDecisionTree.findSplitsBins(featuress, fakemetadata) //todo: make it so I don't need to give this an RDD[Vector]

    val rawFeatureSplitsAndBins = rawFeatureSplits zip rawFeatureBins
    val featureSplitsAndBins = for(i <- (0 until nOffsets).toArray; sb <- rawFeatureSplitsAndBins) yield {
      val rawSplits = sb._1
      val rawBins = sb._2

      val allRawSplits = rawSplits ++ Seq(rawBins.head.lowSplit, rawBins.last.highSplit)
      val allRawSplitToSplit = allRawSplits.map(s => s ->
        s.copy(feature = s.feature + i*nBaseFeatures)
      ).toMap

      val splits = rawSplits.map(allRawSplitToSplit(_))
      val bins = rawBins.map(b =>
        b.copy(lowSplit = allRawSplitToSplit(b.lowSplit), highSplit = allRawSplitToSplit(b.highSplit))
      )

      (splits, bins)
    }

    val featureSplits = featureSplitsAndBins.map(_._1)
    val featureBins = featureSplitsAndBins.map(_._2)

    println(" done...")
    (featureSplits, featureBins)
  }
}
