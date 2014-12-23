package org.apache.spark.mllib.tree.loss

import main.scala.org.apache.spark.mllib.tree.model.MyModel
import org.apache.spark.mllib.tree.{NeuronUtils, Indexer3D, Double3}
import org.apache.spark.mllib.tree.impl.MyTreePoint
import org.apache.spark.mllib.tree.model.MyTreeEnsembleModel
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object MalisLoss extends MyLoss {

  def gradient(model: MyModel,
               points: RDD[MyTreePoint]): RDD[(MyTreePoint, Double3)] = {
    val preds = model.predict(points.map(_.getFeatureVector))

    val g = points.zip(preds).mapPartitions(k => {
      grad(k.toArray).toIterator
    })
    g
  }

  def segment(model: MyModel,
               points: RDD[MyTreePoint]): RDD[(Int, Int)] = {
    val preds = model.predict(points.map(_.getFeatureVector))

    val s = points.zip(preds).mapPartitions(k => {
      seg(k.toArray).toIterator
    }).cache()
    s
  }

  def computeError(model: MyModel, data: RDD[MyTreePoint]): Double = ???

  case class Edge(point:MyTreePoint, weight:Double, from:Int, to:Int, dir:Int)

  def grad(pointsAndPreds: Array[(MyTreePoint, Double3)]) = {
    val subvolume_size = 20
    val dimensions = pointsAndPreds(0)._1.data.dimensions

    var dMax = Int.MaxValue //todo:remove
    var df:Iterable[Int] = null //
    var dt:Iterable[Int] = null //

    val submaps = for(x <- 0 to dimensions._1/subvolume_size;
        y <- 0 to dimensions._2/subvolume_size;
        z <- 0 to dimensions._3/subvolume_size) yield {
      val minIdx = (x * subvolume_size, y * subvolume_size, z * subvolume_size)
      val max_x = math.min((x+1) * subvolume_size - 1, dimensions._1 - 1)
      val max_y = math.min((y+1) * subvolume_size - 1, dimensions._2 - 1)
      val max_z = math.min((z+1) * subvolume_size - 1, dimensions._3 - 1)
      val maxIdx = (max_x, max_y, max_z)
      val indexer = new Indexer3D(dimensions, minIdx, maxIdx)
      val (grads, d, df_sub, dt_sub) = gradForSubvolume(pointsAndPreds, indexer)
      if(d < dMax) {
        df = df_sub
        dt = dt_sub
        dMax = d
      }
      grads
    }

    println("\n\nBEST delta WAS " + dMax + "\n\n")
//    NeuronUtils.saveSeg("/home/luke/Documents/asdf1", df.map(_ -> 0).toIterator)
//    NeuronUtils.saveSeg("/home/luke/Documents/asdf2", dt.map(_ -> 0).toIterator)
    submaps.reduce(_ ++ _)
  }

  def seg(pointsAndPreds: Array[(MyTreePoint, Double3)]): Seq[(Int, Int)] = {
    val subvolume_size = 20
    val dimensions = pointsAndPreds(0)._1.data.dimensions

    val submaps = for(x <- 0 to dimensions._1/subvolume_size;
                      y <- 0 to dimensions._2/subvolume_size;
                      z <- 0 to dimensions._3/subvolume_size) yield {
      val minIdx = (x * subvolume_size, y * subvolume_size, z * subvolume_size)
      val max_x = math.min((x+1) * subvolume_size - 1, dimensions._1 - 1)
      val max_y = math.min((y+1) * subvolume_size - 1, dimensions._2 - 1)
      val max_z = math.min((z+1) * subvolume_size - 1, dimensions._3 - 1)
      val maxIdx = (max_x, max_y, max_z)
      val indexer = new Indexer3D(dimensions, minIdx, maxIdx)
      val seg = segForSubvolume(pointsAndPreds, indexer)
      val idxs_segs = (0 until indexer.size).map(indexer.innerToOuter).zip(seg)
      idxs_segs
    }
    submaps.reduce(_ ++ _)
  }

  def segForSubvolume(pointsAndPreds: Array[(MyTreePoint, Double3)], indexer:Indexer3D):Array[Int] = {
    println("Seg for subvolume: " + indexer.minIdx + " - " + indexer.maxIdx)
    val pointsAndLabels = pointsAndPreds.map{case (p, a) => (p, p.label)} //todo: this is a complete waste

    val genSeg: Int => Int = kruskals(pointsAndLabels, indexer, edgeFilter = _.weight == 1)._3
    val seg = Array.tabulate[Int](indexer.size)(genSeg)
    seg
  }

  def gradForSubvolume(pointsAndPreds: Array[(MyTreePoint, Double3)], indexer:Indexer3D) = {
    val seg = segForSubvolume(pointsAndPreds, indexer)

    println("Grad for subvolume: " + indexer.minIdx + " - " + indexer.maxIdx)
    val gradients = new ArrayBuffer[(MyTreePoint, Double3)]()

    var e:Edge = null //todo: all this shit can go once I've got good pictures
    var c:Array[IndexedSeq[Int]] = null
    var af:Int = 0
    var at:Int = 0
    var df:Map[Int, IndexedSeq[Int]] = null
    var dt:Map[Int, IndexedSeq[Int]] = null
    var d:Int = Int.MaxValue

    def innerFunc(edge:Edge, children:Array[IndexedSeq[Int]], ancFrom:Int, ancTo:Int) = {
      //println("Adding wedge with weight " + edge.weight)
      def getDescendants(nodeIdx:Int):IndexedSeq[Int] = children(nodeIdx).flatMap(getDescendants) :+ nodeIdx

      val descFrom = getDescendants(ancFrom).groupBy(seg)
      val descTo = getDescendants(ancTo).groupBy(seg)
      val del = ( for((segFrom, subsetFrom) <- descFrom;
                      (segTo, subsetTo) <- descTo) yield
          subsetFrom.size * subsetTo.size * (if(segFrom == segTo && segFrom != 0) 1 else -1)
        ).reduce(_+_)

      gradients.append(edge.point -> (edge.dir match {
        case 1 => Double3(del, 0, 0)
        case 2 => Double3(0, del, 0)
        case 3 => Double3(0, 0, del)
      }))

      if(del < d) {
        e = edge
        c = children
        af = ancFrom
        at = ancTo
        df = descFrom
        dt = descTo
        d = del
      }
    }

    kruskals(pointsAndPreds, indexer, innerFunc = innerFunc)

    val df_out = df.flatMap(_._2.map(indexer.innerToOuter))
    val dt_out = dt.flatMap(_._2.map(indexer.innerToOuter))

    val grads = gradients.groupBy(_._1).map{ case (p, s) =>
      p -> s.map(_._2).reduce(_+_)
    }

    (grads, d, df_out, dt_out)
  }


  def doNothing(e:Edge, a:Array[IndexedSeq[Int]], i1:Int, i2:Int) = {}

  def kruskals(pointsAndAffs: Array[(MyTreePoint, Double3)], indexer:Indexer3D, edgeFilter:Edge => Boolean = _ => true,
               innerFunc:(Edge, Array[IndexedSeq[Int]], Int, Int) => Unit = doNothing)
  :(Array[IndexedSeq[Int]], Array[Int], Int => Int) = {

    val edgeList = (0 until indexer.size).flatMap( i => { // i is an INNER index
      val (point, affs) = pointsAndAffs(indexer.innerToOuter(i))
      val multi = indexer.innerToMulti(i)
      val l1 = if(multi._1 >= indexer.innerDimensions._1-1) List()
               else List(Edge(point, affs._1, i, i + indexer.innerSteps._1, 1))

      val l2 = if(multi._2 >= indexer.innerDimensions._2-1) List()
               else List(Edge(point, affs._2, i, i + indexer.innerSteps._2, 2))

      val l3 = if(multi._3 >= indexer.innerDimensions._3-1) List()
               else List(Edge(point, affs._3, i, i + indexer.innerSteps._3, 3))

      l1 ++ l2 ++ l3
    })

    val edges = new mutable.PriorityQueue[Edge]()(Ordering.by(_.weight)) //Order by weight DESCENDING
    edges ++= edgeList.filter(edgeFilter)

    val parent = Array.tabulate[Int](indexer.size)(i => i)
    val children = Array.fill[IndexedSeq[Int]](indexer.size){ IndexedSeq[Int]() }
    def getAncestor(nodeIdx:Int):Int = {
        if(parent(nodeIdx) == nodeIdx) nodeIdx else getAncestor(parent(nodeIdx))
    }

    var numEdges = 0
    while(numEdges < indexer.size-1 && edges.nonEmpty) {
      val edge = edges.dequeue()
      val ancFrom = getAncestor(edge.from)
      val ancTo = getAncestor(edge.to)
      if(ancFrom != ancTo) {
        innerFunc(edge, children, ancFrom, ancTo)
        if(children(ancFrom).size > children(ancTo).size) {
          parent(ancTo) = ancFrom
          children(ancFrom) = children(ancFrom) :+ ancTo
        } else {
          parent(ancFrom) = ancTo
          children(ancTo) = children(ancTo) :+ ancFrom
        }
        numEdges = numEdges + 1
      }
    }

    (children, parent, getAncestor)
  }
}


