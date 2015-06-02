package org.apache.spark.mllib.tree

import java.awt.Color
import java.awt.image.{Raster, BufferedImage}
import java.io
import java.io.{File, RandomAccessFile, FileWriter}
import java.nio.{FloatBuffer, ByteBuffer}
import java.text.SimpleDateFormat
import java.util.Date
import javax.imageio.ImageIO

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
    println(new SimpleDateFormat("HH:mm:ss").format(new Date()) + " ...complete.")
//    if(sc.getRDDStorageInfo.length == nCached)
//      throw new Exception("Did not have enough memory to cache " + rdd + "! Failing.")

    def unpersist() = {
      println(new SimpleDateFormat("HH:mm:ss").format(new Date()) + " Uncaching " + rdd)
      cachedRDD.unpersist()
    }
    (newRDD, _ => unpersist())
  }

  def getSplitsAndBins(subvolumes: Seq[String], nBaseFeatures:Int, data_root:String, maxBins:Int, numOffsets:Int) = {
    println("getting splits and bins")
    val features_file_1 = data_root + "/" + subvolumes(0) + "/features.raw"
    val features_data_1 = new RawFeatureData(subvolumes(0), features_file_1, nBaseFeatures)
    getSplitsAndBinsFromFeaturess(features_data_1.toVectors.take(100000).toArray, maxBins, nBaseFeatures, numOffsets)//todo SORT THIS VECTOR ITERATOR ARRAY NONSENSE
  }

  def loadData(sc: SparkContext, numExecutors:Int, subvolumes: Seq[String], nBaseFeatures: Int, data_root: String,
               maxBins:Int, offsets:Seq[(Int, Int, Int)], offsetMultiplier:Array[Int], proportion: Double,
               bins:Array[Array[Bin]], fromFront: Boolean) = {
    val chunked = subvolumes.grouped((subvolumes.size + numExecutors - 1) / numExecutors).toSeq

    val rawFeaturesData = sc.parallelize(chunked, chunked.size).mapPartitions{
      _.next().map{ s =>
        new RawFeatureData(s, data_root + "/" + s + "/features.raw", nBaseFeatures)
      }.toIterator
    }

    val dimensions = getDimensions(sc, data_root, chunked, proportion, fromFront)
    val data = rawFeaturesData.zip(dimensions).mapPartitions{p =>

      val startTime = System.currentTimeMillis()

      val d = p.flatMap { case (rawData, dimensions) =>
        val targets = getTargets(data_root, rawData.id, dimensions.n_targets, dimensions.target_index_offset, proportion, fromFront)

        val indexer = new Indexer(dimensions.outerDimensions, dimensions.min_idx, dimensions.max_idx)

        val binnedFeatureData = new BinnedFeatureData(rawData, bins, indexer, offsets, offsetMultiplier)
        targets.zipWithIndex.map { case (ts, idx) =>
          val y = DoubleTuple(ts)
          //val seg = ts(2).toInt
          val seg = 0
          val outer_idx = indexer.innerToOuter(idx)
          new MyTreePoint(y, seg, binnedFeatureData, idx, outer_idx)
        }
      }
      println("creating partition data took " + (System.currentTimeMillis() - startTime) + " ms")
      d
    }
    (data, dimensions.mapPartitions{ p => Seq(p.toArray).toIterator}.collect.toArray)
  }


  /*def randomLabeledData1D(sc: SparkContext, subvolumes: Seq[String], nBaseFeatures: Int, data_root: String,
                          proportion: Double, fromFront: Boolean) = {
    println("USING COMPLETELY RANDOM SYNTHETIC DATA")
    val data = sc.parallelize(1 to subvolumes.size, subvolumes.size).mapPartitionsWithIndex((i, _) => {
      Array.tabulate(1000000){_ =>
        new LabeledPoint(Math.random(), Vectors.dense(Array.fill(nBaseFeatures)(math.random)))
      }.toIterator
    }).cache()

    val dimensions = Array.tabulate(subvolumes.size){_ =>
      Dimensions((100, 100, 100), (0, 0, 0), (199, 199, 199), 1000000, 0)
    }
    (data, dimensions)
  }*/

  case class Dimensions(outerDimensions:(Int, Int, Int), min_idx:(Int, Int, Int), max_idx:(Int, Int, Int), n_targets:Int, target_index_offset:Int)
  private def getDimensions(sc:SparkContext, data_root:String, subvolumes_chunked:Seq[Seq[String]], proportion:Double, fromFront:Boolean) = {
    val numExecutors = sc.getExecutorStorageStatus.length

    sc.parallelize(subvolumes_chunked, subvolumes_chunked.length).mapPartitions(_.next().map{ subvolume => {
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
    }}.toIterator)
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


  def saveText(path:String, filename:String, text:String): Unit = {
    println("Saving Text: " + path + "/" + filename)
    val dir = new io.File(path)
    if (!dir.exists) dir.mkdirs()

    val fwdims = new FileWriter(path + "/" + filename, false)
    fwdims.write(text)
    fwdims.close()
  }

  def save2D(path:String, filename:String, that:Array[Double], dims:(Int, Int)): Unit = {
    println("Saving 2D: " + path + "/" + filename)
    val dir =  new io.File(path)
    if(!dir.exists) dir.mkdirs()

    val fwdims = new FileWriter(path + "/dims.txt", false)
    fwdims.write(dims._1 + " " + dims._2)
    fwdims.close()

    val vals = that.map(x => grayToRGB((x * 255).toInt))
    val img = new BufferedImage(dims._2, dims._1, BufferedImage.TYPE_BYTE_GRAY)
    img.setRGB(0, 0, dims._2, dims._1, vals, 0, dims._2)
    ImageIO.write(img, "png", new File(path + "/" + filename + ".png"))
  }

  def save3D(path:String, filename:String, that:Array[Double], dims:(Int, Int, Int)): Unit = {
    println("Saving 3D: " + path + "/" + filename)
    val dir =  new io.File(path)
    if(!dir.exists) dir.mkdirs()

    val fwdims = new FileWriter(path + "/dims.txt", false)
    fwdims.write(dims._1 + " " + dims._2 + " " + dims._3)
    fwdims.close()

    val fc = new RandomAccessFile(path + "/" + filename, "rw").getChannel
    val byteBuffer = ByteBuffer.allocate(4 * 1) //must be multiple of 4 for floats
    val floatBuffer =  byteBuffer.asFloatBuffer()
    that.foreach { d =>
      floatBuffer.put(d.toFloat)
      fc.write(byteBuffer)
      byteBuffer.rewind()
      floatBuffer.clear()
    }
    fc.close()
  }

  def grayToRGB(x:Int) = {
    val y = Math.min(255, Math.max(x, 0))
    new Color(y, y, y).getRGB
  }

  def saveLabelsAndPredictions(path:String, labelsAndPredictions:Iterator[(DoubleTuple, DoubleTuple, Int /*inner_idx*/)], dimensions:Dimensions,
                               description:String, training_time:Long): Unit = {
    println("Saving labels and predictions: " + path)
    val dir =  new io.File(path)
    if(!dir.exists) dir.mkdirs()

    val fwdescription = new FileWriter(path + "/description.txt", false)
    fwdescription.write(description)
    fwdescription.write("\nTraining took " + training_time + " minutes.")
    fwdescription.close()

    val fwdimensions = new FileWriter(path + "/dimensions.txt", false)
    import dimensions.{min_idx, max_idx}
    val dims = (max_idx._1 - min_idx._1 + 1, max_idx._2 - min_idx._2 + 1, max_idx._3 - min_idx._3 + 1)
    fwdimensions.write(dims._1 + " " + dims._2 + " " + dims._3)
    fwdimensions.close()

    val fclabels = new RandomAccessFile(path + "/labels.raw", "rw").getChannel //todo can use save3d
    val fcpredictions = new RandomAccessFile(path + "/predictions.raw", "rw").getChannel
    val byteBuffer = ByteBuffer.allocate(4 * 3) //must be multiple of 4 for floats
    val floatBuffer =  byteBuffer.asFloatBuffer()
    labelsAndPredictions.foreach{ case (label, prediction, _) =>
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


    println("\tComplete")
  }
}
