#!/usr/bin/env python3
"""
compliant_follow.py — 外力で手で動かせる (擬似コンプライアンス) モード

位置制御のアームでも、「目標 = 現在位置」へ追従させ続けることで、
手で押した方向に腕が付いてくる挙動を実現する。

  - 各関節の 目標(cmd) と 実測(q) のずれが --deadband[rad] を超えたら
    外力が加わったと判断し、目標を実測側へ --gain の割合で寄せる。
  - 外力が無いとき(ずれが小さい)は目標を保持 → その場で止まる(重力で垂れにくい)。
  - Ctrl-C で終了 (最後の位置を保持して終わる)。

注意: トルクを抜くわけではないので、強く押すとモータ負荷が大きくなる。
      ゆっくり押すこと。動きすぎる/重い場合は --deadband, --gain を調整する。

実行例:
  ./run_task.sh compliant_follow.py
  ./run_task.sh compliant_follow.py --deadband 0.015 --gain 0.8
"""
import argparse

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINTS = ['joint1', 'joint2', 'joint3', 'joint4']


class CompliantFollow(Node):
    def __init__(self, topic, deadband, gain, rate, max_step):
        super().__init__('compliant_follow')
        self.deadband = deadband
        self.gain = gain
        self.max_step = max_step
        self.q = None
        self.cmd = None
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)
        self.pub = self.create_publisher(JointTrajectory, topic, 10)
        self.dt = 1.0 / rate
        self.create_timer(self.dt, self._loop)

    def _js_cb(self, m):
        if not set(JOINTS).issubset(m.name):
            return
        self.q = np.array([m.position[m.name.index(n)] for n in JOINTS])
        if self.cmd is None:
            self.cmd = self.q.copy()
            self.get_logger().info(f'初期位置を保持: {np.round(self.cmd, 3)}')

    def _loop(self):
        if self.q is None:
            return
        err = self.q - self.cmd
        # デッドバンド外のずれのみ「外力」とみなして目標を寄せる
        push = np.where(np.abs(err) > self.deadband, err, 0.0)
        step = np.clip(self.gain * push, -self.max_step, self.max_step)
        self.cmd = self.cmd + step
        self._send(self.cmd)

    def _send(self, q):
        m = JointTrajectory()
        m.joint_names = JOINTS
        p = JointTrajectoryPoint()
        p.positions = [float(x) for x in q]
        t = self.dt * 3  # 短い補間時間で滑らかに追従
        p.time_from_start = Duration(sec=0, nanosec=int(t * 1e9))
        m.points = [p]
        self.pub.publish(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/arm_controller/joint_trajectory')
    ap.add_argument('--deadband', type=float, default=0.02, help='外力判定のずれ閾値[rad] (小さいほど軽く動く)')
    ap.add_argument('--gain', type=float, default=0.7, help='目標を実測へ寄せる割合 0-1')
    ap.add_argument('--rate', type=float, default=30.0, help='制御周期[Hz]')
    ap.add_argument('--max-step', type=float, default=0.05, help='1周期あたりの最大目標更新量[rad]')
    a = ap.parse_args()

    rclpy.init()
    node = CompliantFollow(a.topic, a.deadband, a.gain, a.rate, a.max_step)
    print('外力追従モード開始。手でアームを押すと動きます。Ctrl-C で終了')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.q is not None:
            node._send(node.q)  # 終了時は現在位置を保持
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
