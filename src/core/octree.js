'use strict';

const THREE = require('three');

/**
 * Note this is not a proper Octree. This is just a one layer octree used for
 * specific purposes.
 */
class Octree {

  constructor() {

    /**
     * @type {Box3}
     */
    this.boudingBox = new THREE.Box3(
      new THREE.Vector3(-1, 0, -3),
      new THREE.Vector3(1, 2, 3)
    );

    /**
     * @type {Vector3}
     */
    this.centerPoint = new THREE.Vector3(0, 0, 0)
      .addVectors(
        this.boudingBox.min,
        this.boudingBox.max,
      ).divideScalar(2);

    this.q0 = [];
    this.q1 = [];
    this.q2 = [];
    this.q3 = [];
    this.q4 = [];
    this.q5 = [];
    this.q6 = [];
    this.q7 = [];
  }

  /**
   * @param vert {Vector3}
   */
  add(vert) {
    const x = this.centerPoint.x;
    const y = this.centerPoint.y;
    const z = this.centerPoint.z;

    if (vert.x >= x) {
      if (vert.y >= y) {
        if (vert.z >= z) {
          this.q0.push(vert);
        } else {
          this.q1.push(vert);
        }
      } else {
        if (vert.z >= z) {
          this.q2.push(vert);
        } else {
          this.q3.push(vert);
        }
      }
    } else {
      if (vert.y >= y) {
        if (vert.z >= z) {
          this.q4.push(vert);
        } else {
          this.q5.push(vert);
        }
      } else {
        if (vert.z >= z) {
          this.q6.push(vert);
        } else {
          this.q7.push(vert);
        }
      }
    }
  }

  /**
   * @param space {Array.<Vector3>}
   */
  approximateConer(space) {
    /**
     * @type {{vert: Vector3, dist: number}}
     */
    const max = space.map(v => {
      return {vert: v, dist: v.distanceToSquared(this.centerPoint)}
    }).reduce((prev, current) => {
      return (prev.dist > current.dist) ? prev : current
    });

    return max.vert;
  }
}

module.exports = Octree;
