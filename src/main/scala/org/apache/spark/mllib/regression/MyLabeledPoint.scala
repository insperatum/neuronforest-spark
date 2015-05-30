/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.regression

import org.apache.spark.SparkException
import org.apache.spark.mllib.linalg.{Vector, Vectors}

import org.apache.spark.mllib.util.NumericParser

import scala.beans.BeanInfo

/**
 * Class that represents the features and labels of a data point.
 *
 * @param label Label for this data point.
 * @param features List of features for this data point.
 */
@BeanInfo
case class MyLabeledPoint(label: Double, features: Vector) { //todo get rid of this class
  override def toString: String = {
    "(%s,%s)".format(label, features)
  }
}

/**
 * Parser for [[org.apache.spark.mllib.regression.MyLabeledPoint]].
 */
object MyLabeledPoint {
  /**
   * Parses a string resulted from `MyLabeledPoint#toString` into
   * an [[org.apache.spark.mllib.regression.MyLabeledPoint]].
   */
  def parse(s: String): MyLabeledPoint = {
    ???
    /*
    if (s.startsWith("(")) {
      NumericParser.parse(s) match {
        case Seq(label: Double, numeric: Any) =>
          MyLabeledPoint(label, Vectors.parseNumeric(numeric))
        case other =>
          throw new SparkException(s"Cannot parse $other.")
      }
    } else { // dense format used before v1.0
      val parts = s.split(',')
      val label = java.lang.Double.parseDouble(parts(0))
      val features = Vectors.dense(parts(1).trim().split(' ').map(java.lang.Double.parseDouble))
      MyLabeledPoint(label, features)
    }*/
  }
}
