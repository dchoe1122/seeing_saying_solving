// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from turtlebot3_msgs:srv/Goal.idl
// generated code does not contain a copyright notice

#ifndef TURTLEBOT3_MSGS__SRV__DETAIL__GOAL__BUILDER_HPP_
#define TURTLEBOT3_MSGS__SRV__DETAIL__GOAL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "turtlebot3_msgs/srv/detail/goal__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace turtlebot3_msgs
{

namespace srv
{


}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::turtlebot3_msgs::srv::Goal_Request>()
{
  return ::turtlebot3_msgs::srv::Goal_Request(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace turtlebot3_msgs


namespace turtlebot3_msgs
{

namespace srv
{

namespace builder
{

class Init_Goal_Response_success
{
public:
  explicit Init_Goal_Response_success(::turtlebot3_msgs::srv::Goal_Response & msg)
  : msg_(msg)
  {}
  ::turtlebot3_msgs::srv::Goal_Response success(::turtlebot3_msgs::srv::Goal_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::turtlebot3_msgs::srv::Goal_Response msg_;
};

class Init_Goal_Response_pose_y
{
public:
  explicit Init_Goal_Response_pose_y(::turtlebot3_msgs::srv::Goal_Response & msg)
  : msg_(msg)
  {}
  Init_Goal_Response_success pose_y(::turtlebot3_msgs::srv::Goal_Response::_pose_y_type arg)
  {
    msg_.pose_y = std::move(arg);
    return Init_Goal_Response_success(msg_);
  }

private:
  ::turtlebot3_msgs::srv::Goal_Response msg_;
};

class Init_Goal_Response_pose_x
{
public:
  Init_Goal_Response_pose_x()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Goal_Response_pose_y pose_x(::turtlebot3_msgs::srv::Goal_Response::_pose_x_type arg)
  {
    msg_.pose_x = std::move(arg);
    return Init_Goal_Response_pose_y(msg_);
  }

private:
  ::turtlebot3_msgs::srv::Goal_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::turtlebot3_msgs::srv::Goal_Response>()
{
  return turtlebot3_msgs::srv::builder::Init_Goal_Response_pose_x();
}

}  // namespace turtlebot3_msgs

#endif  // TURTLEBOT3_MSGS__SRV__DETAIL__GOAL__BUILDER_HPP_
