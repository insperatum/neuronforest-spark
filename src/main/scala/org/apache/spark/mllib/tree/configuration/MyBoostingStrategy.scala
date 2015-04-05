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

package org.apache.spark.mllib.tree.configuration

import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.loss.{MalisLoss, LogLoss, MyLoss, SquaredError}

import scala.beans.BeanProperty

/**
 * :: Experimental ::
 * Configuration options for [[org.apache.spark.mllib.tree.GradientBoostedTrees]].
 *
 * @param forestStrategy Parameters for the tree algorithm. We support regression and binary
 *                     classification for boosting. Impurity setting will be ignored.
 * @param loss MyLoss function used for minimization during gradient boosting.
 * @param numIterations Number of iterations of boosting.  In other words, the number of
 *                      weak hypotheses used in the final model.
 * @param learningRate Learning rate for shrinking the contribution of each estimator. The
 *                     learning rate should be between in the interval (0, 1]
 */
@Experimental
case class MyBoostingStrategy(
    // Required boosting parameters
    @BeanProperty var forestStrategy: MyStrategy,
    @BeanProperty var loss: MyLoss,
    // Optional boosting parameters
    @BeanProperty var initialTrees: Int = 1,
    @BeanProperty var numIterations: Int = 50,
    @BeanProperty var treesPerIteration: Int = 1,
    @BeanProperty var learningRate: Double = 0.1,
    @BeanProperty var momentum: Double = 0) extends Serializable {

  /**
   * Check validity of parameters.
   * Throws exception if invalid.
   */
  private[tree] def assertValid(): Unit = {
    forestStrategy.algo match {
      case Classification =>
        require(forestStrategy.numClasses == 2,
          "Only binary classification is supported for boosting.")
      case Regression =>
        // nothing
      case _ =>
        throw new IllegalArgumentException(
          s"BoostingStrategy given invalid algo parameter: ${forestStrategy.algo}." +
            s"  Valid settings are: Classification, Regression.")
    }
//    require(learningRate > 0 && learningRate <= 1,
//      "Learning rate should be in range (0, 1]. Provided learning rate is " + s"$learningRate.")
  }
}

@Experimental
object MyBoostingStrategy {

  /**
   * Returns default configuration for the boosting algorithm
   * @param algo Learning goal.  Supported:
   *             [[org.apache.spark.mllib.tree.configuration.Algo.Classification]],
   *             [[org.apache.spark.mllib.tree.configuration.Algo.Regression]]
   * @return Configuration for boosting algorithm
   */
  def defaultParams(algo: String): MyBoostingStrategy = {
    val treeStrategy = MyStrategy.defaultStrategy(algo)
    treeStrategy.maxDepth = 3
    algo match {
      /*case "Classification" =>
        treeStrategy.numClasses = 2
        new MyBoostingStrategy(treeStrategy, LogLoss)*/
      case "Regression" =>
        //new MyBoostingStrategy(treeStrategy, SquaredError)
        new MyBoostingStrategy(treeStrategy, MalisLoss)
      case _ =>
        throw new IllegalArgumentException(s"$algo is not supported by boosting.")
    }
  }
}
