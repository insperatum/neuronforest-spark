package org.apache.spark.mllib.tree

import java.io
import java.io.{RandomAccessFile, FileWriter}
import java.nio.{FloatBuffer, ByteBuffer}
import java.text.SimpleDateFormat
import java.util.Date

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.{MyStrategy, Strategy}
import org.apache.spark.mllib.tree.configuration.QuantileStrategy._
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impl.{MyDecisionTreeMetadata, MyTreePoint}
import org.apache.spark.mllib.tree.impurity.{MyVariance, Gini}
import org.apache.spark.mllib.tree.model.{Bin, Split}

import org.apache.spark.rdd.RDD

import scala.io.Source
import scala.reflect.ClassTag

object NeuronUtils {

  def cached[T: ClassTag](rdd:RDD[T]): (RDD[T], Unit => Unit) = { //todo: add unpersist!
    println(new SimpleDateFormat("HH:mm:ss").format(new Date()) + " Caching " + rdd)
    val sc = rdd.sparkContext
    val nCached = sc.getRDDStorageInfo.length
    val cachedRDD = rdd.mapPartitions(p =>
      Iterator(p.toSeq)
    )
    cachedRDD.cache()

    val newRDD = cachedRDD.mapPartitions(p =>
      p.next().toIterator
    )
    cachedRDD.count() // force computation

    if(sc.getRDDStorageInfo.length == nCached)
      throw new Exception("Did not have enough memory to cache " + rdd + "! Failing.")

    def unpersist() = {
      println(new SimpleDateFormat("HH:mm:ss").format(new Date()) + " Uncaching " + rdd)
      cachedRDD.unpersist()
    }
    (newRDD, _ => unpersist())
  }

  def getSplitsAndBins(subvolumes: Seq[String], nBaseFeatures:Int, data_root:String, maxBins:Int, offsets:Seq[(Int, Int, Int)]) = {
    println("getting splits and bins")
    val features_file_1 = data_root + "/" + subvolumes(0) + "/features.raw"
    val features_data_1 = new RawFeatureData(features_file_1, nBaseFeatures)
    getSplitsAndBinsFromFeaturess(features_data_1.toVectors.take(100000).toArray, maxBins, nBaseFeatures, offsets.length)//todo SORT THIS VECTOR ITERATOR ARRAY NONSENSE
  }

  def loadData(sc: SparkContext, subvolumes: Seq[String], nBaseFeatures: Int, data_root: String,
               maxBins:Int, offsets:Seq[(Int, Int, Int)], proportion: Double, bins:Array[Array[Bin]], fromFront: Boolean) = {
    val rawFeaturesData = sc.parallelize(1 to subvolumes.size, subvolumes.size).mapPartitionsWithIndex((i, _) => {
      val features_file = data_root + "/" + subvolumes(i) + "/features.raw"
      Seq(new RawFeatureData(features_file, nBaseFeatures)).toIterator
    })

    val dimensions = getDimensions(sc, data_root, subvolumes, proportion, fromFront)
    val data = rawFeaturesData.zip(dimensions).mapPartitionsWithIndex((i, p) => {
      val startTime = System.currentTimeMillis()

      val (rawData, dimensions) = p.next()

      val targets = getTargets(data_root, subvolumes(i), dimensions.n_targets, dimensions.target_index_offset, proportion, fromFront)

      val indexer = new Indexer3D(dimensions.outerDimensions, dimensions.min_idx, dimensions.max_idx)

      val binnedFeatureData = new BinnedFeatureData(rawData, bins, indexer, offsets)
      val d = targets.zipWithIndex.map { case (ts, idx) =>
        val y = Double3(ts(0), ts(1), ts(2))
        val seg = ts(3).toInt
        val outer_idx = indexer.innerToOuter(idx)
        new MyTreePoint(y, seg, binnedFeatureData, idx, outer_idx)
      }

      println("creating partition data took " + (System.currentTimeMillis() - startTime) + " ms")
      d
    })
    (data, dimensions.collect())
  }



  def loadLabeledData1D(sc: SparkContext, subvolumes: Seq[String], nBaseFeatures: Int, data_root: String,
               proportion: Double, fromFront: Boolean) = {
    val rawFeaturesData = sc.parallelize(1 to subvolumes.size, subvolumes.size).mapPartitionsWithIndex((i, _) => {
      val features_file = data_root + "/" + subvolumes(i) + "/features.raw"
      Seq(new RawFeatureData(features_file, nBaseFeatures)).toIterator
    })

    val dimensions = getDimensions(sc, data_root, subvolumes, proportion, fromFront)
    val data = rawFeaturesData.zip(dimensions).mapPartitionsWithIndex((i, p) => {
      val startTime = System.currentTimeMillis()

      val (rawData, dimensions) = p.next()

      val targets = getTargets(data_root, subvolumes(i), dimensions.n_targets, dimensions.target_index_offset, proportion, fromFront)

      val indexer = new Indexer3D(dimensions.outerDimensions, dimensions.min_idx, dimensions.max_idx)

      val featureVectors = rawData.toVectors.toArray

      val d = targets.zipWithIndex.map { case (ts, idx) =>
        val y = Double3(ts(0), ts(1), ts(2))
        val seg = ts(3).toInt
        val outer_idx = indexer.innerToOuter(idx)
        //new MyTreePoint(y, seg, binnedFeatureData, idx, outer_idx)
        val features = featureVectors(outer_idx)
        new LabeledPoint(ts(0), features)
      }

      println("creating partition data took " + (System.currentTimeMillis() - startTime) + " ms")
      d
    })
    (data, dimensions.collect())
  }

  case class Dimensions(outerDimensions:(Int, Int, Int), min_idx:(Int, Int, Int), max_idx:(Int, Int, Int), n_targets:Int, target_index_offset:Int)

  private def getDimensions(sc:SparkContext, data_root:String, subvolumes:Seq[String], proportion:Double, fromFront:Boolean) = {
    sc.parallelize(subvolumes, subvolumes.length).map(subvolume => {
      val dimensions_file = data_root + "/" + subvolume + "/dimensions.txt"
      val dimensions = Source.fromFile(dimensions_file).getLines().map(_.split(" ").map(_.toInt)).toArray

      val outerDimensions = (dimensions(0)(0), dimensions(0)(1), dimensions(0)(2))
      //val size = dimensions(0)
      //val step = (size(1) * size(2), size(2), 1)
      val min_idx_all = (dimensions(1)(0), dimensions(1)(1), dimensions(1)(2))
      val max_idx_all = (dimensions(2)(0), dimensions(2)(1), dimensions(2)(2))

      val min_idx = if(fromFront) min_idx_all
        else (max_idx_all._1 - ((max_idx_all._1 - min_idx_all._1)*proportion).toInt, min_idx_all._2, min_idx_all._3)

      val max_idx = if(!fromFront) max_idx_all
        else (min_idx_all._1 + ((max_idx_all._1 - min_idx_all._1)*proportion).toInt, max_idx_all._2, max_idx_all._3)

      val n_targets = (max_idx._1 - min_idx._1 + 1) * (max_idx._2 - min_idx._2 + 1) * (max_idx._3 - min_idx._3 + 1)
      val target_index_offset = (min_idx._1 - min_idx_all._1) * (max_idx._2 - min_idx._2 + 1) * (max_idx._3 - min_idx._3 + 1)

      Dimensions(outerDimensions, min_idx, max_idx, n_targets, target_index_offset)
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
    val strategy = new MyStrategy(Classification, MyVariance, 0, 0, maxBins, Sort, Map[Int, Int]())
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


  def saveSeg(path:String, filename:String, seg:Array[Int]): Unit = {
    println("Saving seg: " + path + "/" + filename)
    val dir =  new io.File(path)
    if(!dir.exists) dir.mkdirs()

    val fcseg = new RandomAccessFile(path + "/" + filename, "rw").getChannel
    val byteBuffer = ByteBuffer.allocate(4) //must be multiple of 4 for ints
    val intBuffer =  byteBuffer.asIntBuffer()
    seg.foreach{ case (s) =>
      intBuffer.put(s)
      fcseg.write(byteBuffer)
      byteBuffer.rewind()
      intBuffer.clear()
    }
    fcseg.close()
  }

  def save3D(path:String, filename:String, that:Array[Double3], dims:(Int, Int, Int)): Unit = {
    println("Saving 3D: " + path + "/" + filename)
    val dir =  new io.File(path)
    if(!dir.exists) dir.mkdirs()

    val fwdims = new FileWriter(path + "/dims.txt", false)
    fwdims.write(dims._1 + " " + dims._2 + " " + dims._3)
    fwdims.close()

    val fc = new RandomAccessFile(path + "/" + filename, "rw").getChannel
    val byteBuffer = ByteBuffer.allocate(4 * 3) //must be multiple of 4 for floats
    val floatBuffer =  byteBuffer.asFloatBuffer()
    that.foreach { d3 =>
      Seq(d3._1, d3._2, d3._3).foreach(d => floatBuffer.put(d.toFloat))
      fc.write(byteBuffer)
      byteBuffer.rewind()
      floatBuffer.clear()
    }
    fc.close()
  }

  def saveLabelsAndPredictions(path:String, labelsAndPredictions:Iterator[(Double3, Double3)], dimensions:Dimensions,
                               description:String, indexesAndGrads:Array[(Int, Double3)] = null): Unit = {
    println("Saving labels and predictions: " + path)
    val dir =  new io.File(path)
    if(!dir.exists) dir.mkdirs()

    val fwdescription = new FileWriter(path + "/description.txt", false)
    fwdescription.write(description)
    fwdescription.close()

    val fwdimensions = new FileWriter(path + "/dimensions.txt", false)
    import dimensions.{min_idx, max_idx}
    val dims = (max_idx._1 - min_idx._1 + 1, max_idx._2 - min_idx._2 + 1, max_idx._3 - min_idx._3 + 1)
    fwdimensions.write(dims._1 + " " + dims._2 + " " + dims._3)
    fwdimensions.close()

    /*val fwlabels = new FileWriter(path + "/labels.txt", false)
    val fwpredictions = new FileWriter(path + "/predictions.txt", false)
    labelsAndPredictions.foreach{ case (label, prediction) => {
      fwlabels.write(label._1 + " " + label._2 + " " + label._3 + "\n")
      fwpredictions.write(prediction._1 + " " + prediction._2 + " " + prediction._3 + "\n")
    }}
    fwlabels.close()
    fwpredictions.close()*/

    val fclabels = new RandomAccessFile(path + "/labels.raw", "rw").getChannel //todo can use save3d
    val fcpredictions = new RandomAccessFile(path + "/predictions.raw", "rw").getChannel
    val byteBuffer = ByteBuffer.allocate(4 * 3) //must be multiple of 4 for floats
    val floatBuffer =  byteBuffer.asFloatBuffer()
    labelsAndPredictions.foreach{ case (label, prediction) =>
      Seq(label._1, label._2, label._3).foreach(d => floatBuffer.put(d.toFloat))
      fclabels.write(byteBuffer)
      byteBuffer.rewind()
      floatBuffer.clear()
      Seq(prediction._1, prediction._2, prediction._3).foreach(d => floatBuffer.put(d.toFloat))
      fcpredictions.write(byteBuffer)
      byteBuffer.rewind()
      floatBuffer.clear()
    }
    fclabels.close()
    fcpredictions.close()


    if(indexesAndGrads != null) {
      val fwgrads = new FileWriter(path + "/gradients.txt", false)
      indexesAndGrads.foreach{ case (idx, grad) =>
        fwgrads.write(idx + " " + grad._1 + " " + grad._2 + " " + grad._3 + "\n")
      }
      fwgrads.close()
    }

    println("\tComplete")
  }
}
