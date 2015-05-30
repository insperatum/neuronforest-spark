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

package org.apache.spark.mllib.tree.loss

import main.scala.org.apache.spark.mllib.tree.model.MyModel
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.regression.MyLabeledPoint
import org.apache.spark.mllib.tree.impl.MyTreePoint
import org.apache.spark.mllib.tree.model.{MyEnsembleModelNew, MyTreeEnsembleModelNew, TreeEnsembleModel}
import org.apache.spark.rdd.RDD

/**
 * :: DeveloperApi ::
 * Trait for adding "pluggable" loss functions for the gradient boosting algorithm.
 */
@DeveloperApi
trait MyLoss extends Serializable {

  def cachedGradientAndLoss(
      model: MyModel,
      points: RDD[MyTreePoint],
      subsample_proportion: Double,
      save_to:String = null): (RDD[(MyTreePoint, Double)], Double, Unit => Unit)

  def computeError(model: MyModel, data: RDD[MyTreePoint]): Double

}
