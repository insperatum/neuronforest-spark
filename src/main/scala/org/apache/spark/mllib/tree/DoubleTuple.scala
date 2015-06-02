package org.apache.spark.mllib.tree

// 2D
object DoubleTuple {
  val size = 3
  def MinValue = DoubleTuple(Double.MinValue, Double.MinValue, Double.MinValue)
  def Zero = DoubleTuple(0, 0, 0)
  def max(a:DoubleTuple, b:DoubleTuple) = DoubleTuple(math.max(a._1, b._1), math.max(a._2, b._2), math.max(a._3, b._3))

  def oneHot(i:Int, d:Double) = i match {
    case 1 => DoubleTuple(d, 0, 0)
    case 2 => DoubleTuple(0, d, 0)
    case 3 => DoubleTuple(0, 0, d)
  }

  def apply(arr:Array[Double]):DoubleTuple = {
    DoubleTuple(arr(0), arr(1), arr(2))
  }
}
case class DoubleTuple(_1:Double, _2:Double, _3:Double) {
  def apply(i: Int) = i match {
    case 1 => _1
    case 2 => _2
    case 3 => _3
  }

  def *(w:Double) = DoubleTuple(w * _1, w * _2, w * _3)
  def /(w:Double) = DoubleTuple(_1 / w, _2 / w, _3 / w)

  def +(x:DoubleTuple) = DoubleTuple(x._1 + _1, x._2 + _2, x._3 + _3)
  def -(x:DoubleTuple) = DoubleTuple(x._1 - _1, x._2 - _2, x._3 - _3)
  def *(x:DoubleTuple) = DoubleTuple(x._1 * _1, x._2 * _2, x._3 * _3)
  def /(x:DoubleTuple) = DoubleTuple(x._1 / _1, x._2 / _2, x._3 / _3)
  def sq = _1 * _1 + _2 * _2 + _3 * _3
  def min = math.min(_1, math.min(_2, _3))
  def max = math.max(_1, math.min(_2, _3))
  def sum = _1 + _2 + _3
  def avg = sum/3
}