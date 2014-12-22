package org.apache.spark.mllib.tree

class Indexer3D(val outerDimensions:(Int, Int, Int), val minIdx:(Int, Int, Int), val maxIdx:(Int, Int, Int)) {
  val innerDimensions = (maxIdx._1 - minIdx._1 + 1, maxIdx._2 - minIdx._2 + 1, maxIdx._3 - minIdx._3 + 1)
  val outerSteps = (outerDimensions._2 * outerDimensions._3, outerDimensions._3, 1)
  val innerSteps = (innerDimensions._2 * innerDimensions._3, innerDimensions._3, 1)
  val size = innerDimensions._1 * innerDimensions._2 * innerDimensions._3

  def innerToOuter(i:Int) =
    outerSteps._1 * (minIdx._1 + i / innerSteps._1) +
    outerSteps._2 * (minIdx._2 + (i % innerSteps._1) / innerSteps._2) +
    outerSteps._3 * (minIdx._3 + i % innerSteps._2)

  def outerToInner(i:Int) =
    innerSteps._1 * (i / outerSteps._1 - minIdx._1) +
    innerSteps._2 * ((i % outerSteps._1) / outerSteps._2 - minIdx._2) +
    innerSteps._3 * (i % outerSteps._2 - minIdx._3)

  def innerToMulti(i:Int) =
    (i / innerSteps._1,
      (i % innerSteps._1) / innerSteps._2,
      (i % innerSteps._2) / innerSteps._3)


}
