- mapをsubscribe
```
create_subscription<nav_msgs::msg::OccupancyGrid>(
	  "map", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable(),
	  std::bind(&EMcl2Node::receiveMap, this, std::placeholders::_1));
```
- mapからA*で経路生成
  - 占有領域を避ける経路
  - click pointが置かれたら一度だけ生成

- rviz2で可視化