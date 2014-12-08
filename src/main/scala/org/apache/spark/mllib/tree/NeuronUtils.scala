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
  def getSplitsAndBins(subvolumes: Array[String], nBaseFeatures:Int, data_root:String, maxBins:Int, offsets:Array[(Int, Int, Int)]) = {
    println("getting splits and bins")
    val features_file_1 = data_root + "/" + subvolumes(0) + "/features.raw"
    val features_data_1 = new RawFeatureData(features_file_1, nBaseFeatures)
    getSplitsAndBinsFromFeaturess(features_data_1.toVectors.take(100000).toArray, maxBins, nBaseFeatures, offsets.length)//todo SORT THIS VECTOR ITERATOR ARRAY NONSENSE
  }

  def loadData(sc: SparkContext, subvolumes: Array[String], nBaseFeatures: Int, data_root: String,
               maxBins:Int, offsets:Array[(Int, Int, Int)], proportion: Double, bins:Array[Array[Bin]], fromFront: Boolean) = {
    val rawFeaturesData = sc.parallelize(1 to subvolumes.size, subvolumes.size).mapPartitionsWithIndex((i, _) => {
      val features_file = data_root + "/" + subvolumes(i) + "/features.raw"
      Seq(new RawFeatureData(features_file, nBaseFeatures)).toIterator
    })
    rawFeaturesData.cache()

    val dimensions = getDimensions(sc, data_root, subvolumes, proportion, fromFront)
    val data = rawFeaturesData.zip(dimensions).mapPartitionsWithIndex((i, p) => {
      val startTime = System.currentTimeMillis()

      val (rawData, dimensions) = p.next()
      import dimensions._
      val targets = getTargets(data_root, subvolumes(i), n_targets, target_index_offset, proportion, fromFront)

      val seg_size = (max_idx._1 - min_idx._1 + 1, max_idx._2 - min_idx._2 + 1, max_idx._3 - min_idx._3 + 1)
      val seg_step = (seg_size._2 * seg_size._3, seg_size._3, 1)

      val binnedFeatureData = new BinnedFeatureData(rawData, bins, seg_step, offsets)
      val d = targets.zipWithIndex.map { case (ts, i) =>
        val t = i + target_index_offset
        val y = ts(0)
        val example_idx =
          step._1 * (min_idx._1 + t / seg_step._1) +
            step._2 * (min_idx._2 + (t % seg_step._1) / seg_step._2) +
            (min_idx._3 + t % seg_step._2)
        new MyTreePoint(y, null, binnedFeatureData, example_idx)
      }

      println("creating partition data took " + (System.currentTimeMillis() - startTime) + " ms")
      d
    })
    rawFeaturesData.unpersist()
    (data, dimensions.collect())
  }



  case class Dimensions(step:(Int, Int, Int), min_idx:(Int, Int, Int), max_idx:(Int, Int, Int), n_targets:Int, target_index_offset:Int)

  private def getDimensions(sc:SparkContext, data_root:String, subvolumes:Array[String], proportion:Double, fromFront:Boolean) = {
    sc.parallelize(subvolumes, subvolumes.length).map(subvolume => {
      val dimensions_file = data_root + "/" + subvolume + "/dimensions.txt"
      val dimensions = Source.fromFile(dimensions_file).getLines().map(_.split(" ").map(_.toInt)).toArray

      val size = dimensions(0)
      val step = (size(1) * size(2), size(2), 1)
      val min_idx_all = (dimensions(1)(0), dimensions(1)(1), dimensions(1)(2))
      val max_idx_all = (dimensions(2)(0), dimensions(2)(1), dimensions(2)(2))

      val min_idx = if(fromFront) min_idx_all
        else (max_idx_all._1 - ((max_idx_all._1 - min_idx_all._1)*proportion).toInt, min_idx_all._2, min_idx_all._3)

      val max_idx = if(!fromFront) max_idx_all
        else (min_idx_all._1 + ((max_idx_all._1 - min_idx_all._1)*proportion).toInt, max_idx_all._2, max_idx_all._3)


      val n_targets = (max_idx._1 - min_idx._1 + 1) * (max_idx._2 - min_idx._2 + 1) * (max_idx._3 - min_idx._3 + 1)
      val target_index_offset = (min_idx._1 - min_idx_all._1) * (max_idx._2 - min_idx._2 + 1) * (max_idx._3 - min_idx._3 + 1)

      println("From " + min_idx + " to " + max_idx)
      Dimensions(step, min_idx, max_idx, n_targets, target_index_offset)
    })
  }


  private def getTargets(data_root:String, subvolume: String, n_targets:Int, target_index_offset:Int, proportion:Double, fromFront:Boolean) = {
    val targets_file = data_root + "/" + subvolume + "/targets.txt"
    val allTargets = Source.fromFile(targets_file).getLines().map(_.split(" ").map(_.toDouble))
    val targets = if (fromFront)
      allTargets.take(n_targets)
    else
      allTargets.drop(target_index_offset)
    targets
  }


  private def getSplitsAndBinsFromFeaturess(featuress:Array[org.apache.spark.mllib.linalg.Vector], maxBins:Int, nBaseFeatures:Int, nOffsets:Int):
  (Array[Array[Split]], Array[Array[Bin]]) = {
    println("getSplitsAndBins")
    val strategy = new Strategy(Classification, Gini, 0, 0, maxBins, Sort, Map[Int, Int]())
    val fakemetadata = MyDecisionTreeMetadata.buildMetadata(featuress(0).size, featuress.size, strategy, 50, "sqrt")
    val (rawFeatureSplits, rawFeatureBins) = MyDecisionTree.findSplitsBins(featuress, fakemetadata)

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
