import org.apache.spark.mllib.tree.Double3

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object MalisLoss {

  case class Edge(weight:Double, from:Int, to:Int, dir:Int) extends Ordered[Edge] {
    override def compare(that: Edge): Int = weight.compare(that.weight)
  }

  def loss(preds:Iterator[Double3], seg:IndexedSeq[Int], dimensions:(Int, Int, Int)) = {
    val n = dimensions._1 * dimensions._2 * dimensions._3
    val steps = (dimensions._2 * dimensions._3, dimensions._3, 1)

    val edgeList = preds.zipWithIndex.flatMap{case (affs, i) =>
      (if(i / steps._1 >= dimensions._1-1)              List() else List(Edge(affs._1, i, i+steps._1, 1))) ++
      (if((i % steps._1) / steps._2 >= dimensions._2-1) List() else List(Edge(affs._1, i, i+steps._2, 2))) ++
      (if((i % steps._2) / steps._3 >= dimensions._3-1) List() else List(Edge(affs._1, i, i+steps._3, 3)))
    }

    val edges = new mutable.PriorityQueue[Edge]
    edges ++= edgeList

    val parent = Array.tabulate[Int](n)(i => i)
    val children = Array.fill[IndexedSeq[Int]](n){ IndexedSeq[Int]() }
    def getAncestor(nodeIdx:Int):Int = if(parent(nodeIdx) == nodeIdx) nodeIdx else getAncestor(parent(nodeIdx))
    def getDescendants(nodeIdx:Int):IndexedSeq[Int] = children(nodeIdx).flatMap(getDescendants) :+ nodeIdx

    var numEdges = 0
    val gradients = new ArrayBuffer[(Edge, Double)]()
    while(numEdges < n-1) {
      val edge = edges.dequeue()
      val ancFrom = getAncestor(edge.from)
      val ancTo = getAncestor(edge.to)
      if(ancFrom != ancTo) {
        val descFrom = getDescendants(ancFrom).groupBy(seg)
        val descTo = getDescendants(ancTo).groupBy(seg)
        println("Joining " + descFrom + " and " + descTo)
        val grad = descFrom.map { case (segFrom, subsetFrom) =>
          descTo.map { case (segTo, subsetTo) =>
            subsetFrom.size * subsetTo.size * (if(segFrom == segTo) 1 else -1)
          }.reduce(_+_)
        }.reduce(_+_)
        gradients.append(edge -> grad)

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

    println("\nDone")
    println("Gradients:" + gradients.map{case (e,d) => e.from + "," + e.to + ": " + d})
  }

  def main(args:Array[String]): Unit = {
    val dims = (2,2,2)
    val n = dims._1 * dims._2 * dims._3
    val preds = (0 until n).iterator.map(i => Double3(Math.random(), Math.random(), Math.random()))
    val seg = (0 until n).map(i => Random.nextInt(2))
    loss(preds, seg, dims)
  }
}


