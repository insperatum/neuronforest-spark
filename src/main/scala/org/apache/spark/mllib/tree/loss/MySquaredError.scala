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
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{NeuronUtils, DoubleTuple}
import org.apache.spark.mllib.tree.impl.MyTreePoint
import org.apache.spark.mllib.tree.model.TreeEnsembleModel
import org.apache.spark.rdd.RDD

/**
 * :: DeveloperApi ::
 * Class for squared error loss calculation.
 *
 * The squared (L2) error is defined as:
 *   (y - F(x))**2
 * where y is the label and F(x) is the model prediction for features x.
 */
@DeveloperApi
object MuSquaredError extends MyLoss {

  /**
   * Method to calculate the gradients for the gradient boosting calculation for least
   * squares error calculation.
   * The gradient with respect to F(x) is: - 2 (y - F(x))
   * @param model Ensemble model
   * @param point Instance of the training dataset
   * @return Loss gradient
   */
//  override def gradient(
//    model: TreeEnsembleModel,
//    point: LabeledPoint): Double = {
//    2.0 * (model.predict(point.features) - point.label)
//  }

  override def cachedGradientAndLoss(
                             model: MyModel,
                             points: RDD[MyTreePoint],
                             subsample_proportion: Double,
                             save_to:String = null): (RDD[(MyTreePoint, DoubleTuple)], Double, Unit => Unit) = {
    val preds = model.predict(points.sample(true, subsample_proportion).map(_.getFeatureVector))

    val (gradAndLoss, uncache) = NeuronUtils.cached(
      points.zip(preds).map{ case (point, pred) =>
        val diff = pred - point.label
        val grad = (point, diff * 2)
        val loss = diff.sq
        (grad, loss)
      }
    )

    val grads = gradAndLoss.map(_._1)
    val loss = gradAndLoss.map(_._2).mean()

    (grads, loss, uncache)
  }


  override def computeError(model: MyModel, data: RDD[MyTreePoint]): Double = ???
}
