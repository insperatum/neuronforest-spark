package main.scala.org.apache.spark.mllib.tree.model

import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

/**
 * Created by luke on 21/12/14.
 */
trait MyModel {
  def predict(features: Vector): Double
  def predict(features: RDD[Vector]): RDD[Double]
  //def predict(features: JavaRDD[Vector]): JavaRDD[Double]
}
