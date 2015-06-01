package org.apache.spark.mllib.tree

import java.io.RandomAccessFile
import java.nio.{ByteBuffer, FloatBuffer}

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.model.Bin
import org.apache.spark.mllib.tree.impl.MyTreePoint

trait FeatureData {
  def getValue(i:Int, f:Int):Double
  def nFeatures:Int
  def nExamples:Int
}

class RawFeatureData(val id:String, file:String, val nFeatures:Int) extends FeatureData with Serializable {
  val arr = loadFeatures(file)//todo: NO!
  val nExamples = arr.length / nFeatures
  def getValue(i:Int, f:Int) = arr(i*nFeatures + f)

  def loadFeatures(path:String):Array[Float] = {
    println("loading raw feature data: " + path)

    val file = new RandomAccessFile(path, "r")
    val fileChannel = file.getChannel

    val byteBuffer = ByteBuffer.allocate(4 * 10000) //must be multiple of 4 for floats
    val outFloatBuffer = FloatBuffer.allocate(fileChannel.size.toInt/4)

    var bytesRead = fileChannel.read(byteBuffer)
    while(bytesRead > 0) {
      byteBuffer.flip()
      outFloatBuffer.put(byteBuffer.asFloatBuffer())
      byteBuffer.clear()
      bytesRead = fileChannel.read(byteBuffer)
    }

    outFloatBuffer.array()
  }

  def toVectors = {
    println("to vectors...")
    val iter = (0 until nExamples).toIterator.map { i =>
      Vectors.dense(arr.view(i, i + nFeatures).map(_.toDouble).toArray) //todo: NO NO!
    }
    iter
  }
}

class BinnedFeatureData(featureData:RawFeatureData,
                        bins:Array[Array[Bin]],
                        val indexer: Indexer2D,
                        offsets:Seq[(Int, Int)],
                        offsetMultiplier:Array[Int]) extends Serializable {


  import indexer.outerSteps
  val dimensions = indexer.innerDimensions //????

  val id = featureData.id
  val arr = featureData.arr
  val binnedBaseFeatures = Array.ofDim[Int](arr.length)
  val nBaseFeatures = featureData.nFeatures
  val nFeatures = nBaseFeatures * offsets.length
  val nExamples = featureData.nExamples
  val featureOffsets = offsets.flatMap(o => {
    val idxOffset = o._1 * outerSteps._1 + o._2 * outerSteps._2
    //Array.fill(nBaseFeatures){idxOffset}
    offsetMultiplier.map(_ * idxOffset)
  })

  var i = 0
  while(i < nExamples) {
    var f = 0
    while(f < nBaseFeatures) {
      val idx = i * nBaseFeatures + f
      binnedBaseFeatures(idx) = MyTreePoint.findBin(arr(idx), 0, false, bins(f))
      f += 1
    }
    i += 1
  }

  def getValue(i:Int, f:Int) = featureData.getValue(i + featureOffsets(f), f % nBaseFeatures)
  def getBin(i:Int, f:Int) = binnedBaseFeatures((i + featureOffsets(f)) * nBaseFeatures + (f % nBaseFeatures))
}