package org.apache.spark.mllib.tree.loss

import org.apache.spark.mllib.tree.{Indexer3D, Double3}
import org.apache.spark.mllib.tree.impl.MyTreePoint
import org.apache.spark.mllib.tree.model.MyTreeEnsembleModel
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object MalisLoss extends MyLoss {

  def gradient(model: MyTreeEnsembleModel,
               points: RDD[MyTreePoint]): RDD[(MyTreePoint, Double3)] = {
    val preds = model.predict(points.map(_.getFeatureVector))
    points.zip(preds).mapPartitions(k => {
      //grad(k.map(_._1).toArray, k.map(_._2).toSeq).toIterator
      grad(k.toArray).toIterator
    })
  }

  def computeError(model: MyTreeEnsembleModel, data: RDD[MyTreePoint]): Double = ???

  case class Edge(point:MyTreePoint, weight:Double, from:Int, to:Int, dir:Int) extends Ordered[Edge] {
    override def compare(that: Edge): Int = weight.compare(that.weight)
  }


  def grad(pointsAndPreds: Array[(MyTreePoint, Double3)]): Map[MyTreePoint, Double3] = {
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
      gradForSubvolume(pointsAndPreds, indexer)
    }
    submaps.reduce(_ ++ _)
  }

  def gradForSubvolume(pointsAndPreds: Array[(MyTreePoint, Double3)], indexer:Indexer3D): Map[MyTreePoint, Double3] = {
    println("Grad for subvolume: " + indexer.minIdx + " - " + indexer.maxIdx)
    def points(i:Int) = pointsAndPreds(i)._1

    val edgeList = (0 until indexer.size).flatMap( i => { // i is an INNER index
      val (point, affs) = pointsAndPreds(indexer.innerToOuter(i))
      val multi = indexer.innerToMulti(i)
      val l1 = if(multi._1 >= indexer.innerDimensions._1-1) List()
               else List(Edge(point, affs._1, i, i + indexer.innerSteps._1, 1))

      val l2 = if(multi._2 >= indexer.innerDimensions._2-1) List()
               else List(Edge(point, affs._2, i, i + indexer.innerSteps._2, 2))

      val l3 = if(multi._3 >= indexer.innerDimensions._3-1) List()
               else List(Edge(point, affs._3, i, i + indexer.innerSteps._3, 3))

      l1 ++ l2 ++ l3
    })

    val edges = new mutable.PriorityQueue[Edge]
    edges ++= edgeList

    val parent = Array.tabulate[Int](indexer.size)(i => i)
    val children = Array.fill[IndexedSeq[Int]](indexer.size){ IndexedSeq[Int]() }
    def getAncestor(nodeIdx:Int):Int = if(parent(nodeIdx) == nodeIdx) nodeIdx else getAncestor(parent(nodeIdx))
    def getDescendants(nodeIdx:Int):IndexedSeq[Int] = children(nodeIdx).flatMap(getDescendants) :+ nodeIdx

    var numEdges = 0
    val gradients = new ArrayBuffer[(MyTreePoint, Double3)]()
    while(numEdges < indexer.size-1) {
      val edge = edges.dequeue()
      val ancFrom = getAncestor(edge.from)
      val ancTo = getAncestor(edge.to)
      if(ancFrom != ancTo) {
        val descFrom = getDescendants(ancFrom).groupBy(i => points(i).seg)
        val descTo = getDescendants(ancTo).groupBy(i => points(i).seg)
        val del = descFrom.map { case (segFrom, subsetFrom) =>
          descTo.map { case (segTo, subsetTo) =>
            subsetFrom.size * subsetTo.size * (if(segFrom == segTo && segFrom != 0) 1 else -1) //todo is this the right behaviour for boundaries?
          }.reduce(_+_)
        }.reduce(_+_)
        gradients.append(edge.point -> (edge.dir match {
          case 1 => Double3(del, 0, 0)
          case 2 => Double3(0, del, 0)
          case 3 => Double3(0, 0, del)
        }))

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

    gradients.groupBy(_._1).map{ case (p, s) =>
      p -> s.map(_._2).reduce(_+_)
    }
  }
}


