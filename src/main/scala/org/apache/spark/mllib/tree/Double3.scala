package org.apache.spark.mllib.tree

object Double3 {
  def MinValue = Double3(Double.MinValue, Double.MinValue, Double.MinValue)
  def Zero = Double3(0, 0, 0)
  def max(a:Double3, b:Double3) = Double3(math.max(a._1, b._1), math.max(a._2, b._2), math.max(a._3, b._3))
}
case class Double3(_1:Double, _2:Double, _3:Double) {
  def *(w:Double) = Double3(w * _1, w * _2, w * _3)
  def /(w:Double) = Double3(_1 / w, _2 / w, _3 / w)

  def +(x:Double3) = Double3(x._1 + _1, x._2 + _2, x._3 + _3)
  def -(x:Double3) = Double3(x._1 - _1, x._2 - _2, x._3 - _3)
  def *(x:Double3) = Double3(x._1 * _1, x._2 * _2, x._3 * _3)
  def /(x:Double3) = Double3(x._1 / _1, x._2 / _2, x._3 / _3)
  def sq = _1 * _1 + _2 * _2 + _3 * _3
}
