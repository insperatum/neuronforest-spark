package org.apache.spark.mllib.tree.loss

import org.apache.spark.mllib.tree.Double3
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
    def points(i:Int) = pointsAndPreds(i)._1

    val dimensions = points(0).data.dimensions
    val n = dimensions._1 * dimensions._2 * dimensions._3
    val steps = (dimensions._2 * dimensions._3, dimensions._3, 1)

    val edgeList = pointsAndPreds.zipWithIndex.flatMap{case ((point, affs), i) => {
      (if(i / steps._1 >= dimensions._1-1)              List() else List(Edge(point, affs._1, i, i+steps._1, 1))) ++
        (if((i % steps._1) / steps._2 >= dimensions._2-1) List() else List(Edge(point, affs._2, i, i+steps._2, 2))) ++
        (if((i % steps._2) / steps._3 >= dimensions._3-1) List() else List(Edge(point, affs._3, i, i+steps._3, 3)))
    }}

    val edges = new mutable.PriorityQueue[Edge]
    edges ++= edgeList

    val parent = Array.tabulate[Int](n)(i => i)
    val children = Array.fill[IndexedSeq[Int]](n){ IndexedSeq[Int]() }
    def getAncestor(nodeIdx:Int):Int = if(parent(nodeIdx) == nodeIdx) nodeIdx else getAncestor(parent(nodeIdx))
    def getDescendants(nodeIdx:Int):IndexedSeq[Int] = children(nodeIdx).flatMap(getDescendants) :+ nodeIdx

    var numEdges = 0
    val gradients = new ArrayBuffer[(MyTreePoint, Double3)]()
    while(numEdges < n-1) {
      if(numEdges%100 == 0) println("Edge " + (numEdges+1) + " of " + n)
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


