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

package org.apache.spark.mllib.tree.model

import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.DoubleTuple
import org.apache.spark.mllib.tree.configuration.FeatureType._

/**
 * :: DeveloperApi ::
 * MyNode in a decision tree.
 *
 * About node indexing:
 *   Nodes are indexed from 1.  MyNode 1 is the root; nodes 2, 3 are the left, right children.
 *   MyNode index 0 is not used.
 *
 * @param id integer node id, from 1
 * @param predict predicted value at the node
 * @param impurity current node impurity
 * @param isLeaf whether the node is a leaf
 * @param split split to calculate left and right nodes
 * @param leftNode  left child
 * @param rightNode right child
 * @param stats information gain stats
 */
@DeveloperApi
class MyNode (
    val id: Int,
    var predict: MyPredict,
    var impurity: Double,
    var isLeaf: Boolean,
    var split: Option[Split],
    var leftNode: Option[MyNode],
    var rightNode: Option[MyNode],
    var stats: Option[MyInformationGainStats]) extends Serializable with Logging {

  override def toString = "id = " + id + ", isLeaf = " + isLeaf + ", predict = " + predict + ", " +
    "impurity =  " + impurity + "split = " + split + ", stats = " + stats

  /**
   * build the left node and right nodes if not leaf
   * @param nodes array of nodes
   */
  @deprecated("build should no longer be used since trees are constructed on-the-fly in training",
    "1.2.0")
  def build(nodes: Array[MyNode]): Unit = {
    logDebug("building node " + id + " at level " + MyNode.indexToLevel(id))
    logDebug("id = " + id + ", split = " + split)
    logDebug("stats = " + stats)
    logDebug("predict = " + predict)
    logDebug("impurity = " + impurity)
    if (!isLeaf) {
      leftNode = Some(nodes(MyNode.leftChildIndex(id)))
      rightNode = Some(nodes(MyNode.rightChildIndex(id)))
      leftNode.get.build(nodes)
      rightNode.get.build(nodes)
    }
  }

  /**
   * predict value if node is not leaf
   * @param features feature value
   * @return predicted value
   */
  def predict(features: Vector, maxDepth:Int = Integer.MAX_VALUE) : DoubleTuple = {
    if (isLeaf || MyNode.indexToLevel(id) >= maxDepth) {
      predict.predict
    } else{
      if (split.get.featureType == Continuous) {
        if (features(split.get.feature) <= split.get.threshold) {
          leftNode.get.predict(features, maxDepth)
        } else {
          rightNode.get.predict(features, maxDepth)
        }
      } else {
        if (split.get.categories.contains(features(split.get.feature))) {
          leftNode.get.predict(features, maxDepth)
        } else {
          rightNode.get.predict(features, maxDepth)
        }
      }
    }
  }

  /**
   * Returns a deep copy of the subtree rooted at this node.
   */
  private[tree] def deepCopy(): MyNode = {
    val leftNodeCopy = if (leftNode.isEmpty) {
      None
    } else {
      Some(leftNode.get.deepCopy())
    }
    val rightNodeCopy = if (rightNode.isEmpty) {
      None
    } else {
      Some(rightNode.get.deepCopy())
    }
    new MyNode(id, predict, impurity, isLeaf, split, leftNodeCopy, rightNodeCopy, stats)
  }

  /**
   * Get the number of nodes in tree below this node, including leaf nodes.
   * E.g., if this is a leaf, returns 0.  If both children are leaves, returns 2.
   */
  private[tree] def numDescendants: Int = if (isLeaf) {
    0
  } else {
    2 + leftNode.get.numDescendants + rightNode.get.numDescendants
  }

  /**
   * Get depth of tree from this node.
   * E.g.: Depth 0 means this is a leaf node.
   */
  private[tree] def subtreeDepth: Int = if (isLeaf) {
    0
  } else {
    1 + math.max(leftNode.get.subtreeDepth, rightNode.get.subtreeDepth)
  }

  /**
   * Recursive print function.
   * @param indentFactor  The number of spaces to add to each level of indentation.
   */
  private[tree] def subtreeToString(indentFactor: Int = 0): String = {

    def splitToString(split: Split, left: Boolean): String = {
      split.featureType match {
        case Continuous => if (left) {
          s"(feature ${split.feature} <= ${split.threshold})"
        } else {
          s"(feature ${split.feature} > ${split.threshold})"
        }
        case Categorical => if (left) {
          s"(feature ${split.feature} in ${split.categories.mkString("{",",","}")})"
        } else {
          s"(feature ${split.feature} not in ${split.categories.mkString("{",",","}")})"
        }
      }
    }
    val prefix: String = " " * indentFactor
    if (isLeaf) {
      prefix + s"Predict: ${predict.predict}\n"
    } else {
      prefix + s"If ${splitToString(split.get, left=true)}\n" +
        leftNode.get.subtreeToString(indentFactor + 1) +
        prefix + s"Else ${splitToString(split.get, left=false)}\n" +
        rightNode.get.subtreeToString(indentFactor + 1)
    }
  }

}

private[tree] object MyNode {

  /**
   * Return a node with the given node id (but nothing else set).
   */
  def emptyNode(nodeIndex: Int): MyNode = new MyNode(nodeIndex, new MyPredict(DoubleTuple.MinValue), -1.0,
    false, None, None, None, None)

  /**
   * Construct a node with nodeIndex, predict, impurity and isLeaf parameters.
   * This is used in `DecisionTree.findBestSplits` to construct child nodes
   * after finding the best splits for parent nodes.
   * Other fields are set at next level.
   * @param nodeIndex integer node id, from 1
   * @param predict predicted value at the node
   * @param impurity current node impurity
   * @param isLeaf whether the node is a leaf
   * @return new node instance
   */
  def apply(
      nodeIndex: Int,
      predict: MyPredict,
      impurity: Double,
      isLeaf: Boolean): MyNode = {
    new MyNode(nodeIndex, predict, impurity, isLeaf, None, None, None, None)
  }

  /**
   * Return the index of the left child of this node.
   */
  def leftChildIndex(nodeIndex: Int): Int = nodeIndex << 1

  /**
   * Return the index of the right child of this node.
   */
  def rightChildIndex(nodeIndex: Int): Int = (nodeIndex << 1) + 1

  /**
   * Get the parent index of the given node, or 0 if it is the root.
   */
  def parentIndex(nodeIndex: Int): Int = nodeIndex >> 1

  /**
   * Return the level of a tree which the given node is in.
   */
  def indexToLevel(nodeIndex: Int): Int = if (nodeIndex == 0) {
    throw new IllegalArgumentException(s"0 is not a valid node index.")
  } else {
    java.lang.Integer.numberOfTrailingZeros(java.lang.Integer.highestOneBit(nodeIndex))
  }

  /**
   * Returns true if this is a left child.
   * Note: Returns false for the root.
   */
  def isLeftChild(nodeIndex: Int): Boolean = nodeIndex > 1 && nodeIndex % 2 == 0

  /**
   * Return the maximum number of nodes which can be in the given level of the tree.
   * @param level  Level of tree (0 = root).
   */
  def maxNodesInLevel(level: Int): Int = 1 << level

  /**
   * Return the index of the first node in the given level.
   * @param level  Level of tree (0 = root).
   */
  def startIndexInLevel(level: Int): Int = 1 << level

  /**
   * Traces down from a root node to get the node with the given node index.
   * This assumes the node exists.
   */
  def getNode(nodeIndex: Int, rootNode: MyNode): MyNode = {
    var tmpNode: MyNode = rootNode
    var levelsToGo = indexToLevel(nodeIndex)
    while (levelsToGo > 0) {
      if ((nodeIndex & (1 << levelsToGo - 1)) == 0) {
        tmpNode = tmpNode.leftNode.get
      } else {
        tmpNode = tmpNode.rightNode.get
      }
      levelsToGo -= 1
    }
    tmpNode
  }

}
