#include "utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// Hesai DataType
struct PandarPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    double timestamp;
    uint16_t ring;                      ///< laser ring number
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(PandarPointXYZIRT,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (double, timestamp, timestamp)
    (uint16_t, ring, ring)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subLaserCloud;   // 雷达消息订阅器
    rclcpp::CallbackGroup::SharedPtr callbackGroupLidar;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloud;      // 没用

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubExtractedCloud;  // 发布有效的雷达点云
    rclcpp::Publisher<lio_sam::msg::CloudInfo>::SharedPtr pubLaserCloudInfo;        // 发布雷达点云信息

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;      // imu消息订阅器
    rclcpp::CallbackGroup::SharedPtr callbackGroupImu;
    std::deque<sensor_msgs::msg::Imu> imuQueue;                         // imu信息缓存器

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdom;   // imu里程计增量订阅器
    rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;
    std::deque<nav_msgs::msg::Odometry> odomQueue;                      // imu里程计信息缓存器

    std::deque<sensor_msgs::msg::PointCloud2> cloudQueue;   // 原始雷达消息缓存器
    sensor_msgs::msg::PointCloud2 currentCloudMsg;          // 当前雷达消息

    // 进行点云偏斜矫正时所需的通过imu积分获得的imu姿态信息
    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<PandarPointXYZIRT>::Ptr tmpPandarCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud;      // 完整点云
    pcl::PointCloud<PointType>::Ptr   extractedCloud; // 偏斜矫正后点云

    int ringFlag = 0;
    int deskewFlag;
    cv::Mat rangeMat; // 点云投影获得的深度图

    // 进行点云偏斜矫正时所需的通过imu里程计获得的imu位置增量
    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lio_sam::msg::CloudInfo cloudInfo;
    double timeScanCur; // 雷达帧扫描开始时间戳
    double timeScanEnd; // 雷达帧扫描结束时间戳
    std_msgs::msg::Header cloudHeader;

    vector<int> columnIdnCountVec;


public:
    ImageProjection(const rclcpp::NodeOptions & options) :
            ParamServer("lio_sam_imageProjection", options), deskewFlag(0)
    {
        callbackGroupLidar = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupImu = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupOdom = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        auto lidarOpt = rclcpp::SubscriptionOptions();
        lidarOpt.callback_group = callbackGroupLidar;
        auto imuOpt = rclcpp::SubscriptionOptions();
        imuOpt.callback_group = callbackGroupImu;
        auto odomOpt = rclcpp::SubscriptionOptions();
        odomOpt.callback_group = callbackGroupOdom;

        subImu = create_subscription<sensor_msgs::msg::Imu>(
            imuTopic, qos_imu,
            std::bind(&ImageProjection::imuHandler, this, std::placeholders::_1),
            imuOpt);
        subOdom = create_subscription<nav_msgs::msg::Odometry>(
            odomTopic + "_incremental", qos_imu,
            std::bind(&ImageProjection::odometryHandler, this, std::placeholders::_1),
            odomOpt);
        subLaserCloud = create_subscription<sensor_msgs::msg::PointCloud2>(
            pointCloudTopic, qos_lidar,
            std::bind(&ImageProjection::cloudHandler, this, std::placeholders::_1),
            lidarOpt);

        pubExtractedCloud = create_publisher<sensor_msgs::msg::PointCloud2>(
            "lio_sam/deskew/cloud_deskewed", 1);
        pubLaserCloudInfo = create_publisher<lio_sam::msg::CloudInfo>(
            "lio_sam/deskew/cloud_info", qos);

        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        tmpPandarCloudIn.reset(new pcl::PointCloud<PandarPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.start_ring_index.assign(N_SCAN, 0);
        cloudInfo.end_ring_index.assign(N_SCAN, 0);

        cloudInfo.point_col_ind.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.point_range.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    // 重置中间变量
    void resetParameters()
    {
        laserCloudIn->clear();    // 雷达点云信息清空
        extractedCloud->clear();  // 偏斜矫正后点云清空
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX)); // 投影深度图

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        // imu通过积分获得的姿态信息
        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection(){}

    void imuHandler(const sensor_msgs::msg::Imu::SharedPtr imuMsg)
    {
        // 将imu信息在雷达坐标系下表达
        sensor_msgs::msg::Imu thisImu = imuConverter(*imuMsg);

        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu); // 缓存imu数据

        if (debugImu) { // debug IMU data 确认惯导的输出是否正常，如果Z轴的加速度是负数，则Z轴反过来了，需要重新标定内外参
            double imuRoll, imuPitch, imuYaw;
            tf2::Quaternion orientation;
            tf2::fromMsg(thisImu.orientation, orientation);
            tf2::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
            std::cout << std::fixed << std::setprecision(6)
                << "IMU time:" << thisImu.header.stamp.sec
                << " acc:" << thisImu.linear_acceleration.x << 
                    "," << thisImu.linear_acceleration.y << 
                    "," << thisImu.linear_acceleration.z
                << " gyro:" << thisImu.angular_velocity.x << 
                    "," << thisImu.angular_velocity.y << 
                    "," << thisImu.angular_velocity.z
                << " roll:" << imuRoll << " pitch:" << imuPitch << " yaw:" << imuYaw << std::endl;
        }
    }

    void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odometryMsg)
    {
        // 缓存imu里程计增量信息
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    void cloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg)
    {
        // 转换雷达点云信息为可处理点云数据并缓存
        if (!cachePointCloud(laserCloudMsg))
            return;

        // 点云偏斜矫正所需的imu数据的预处理(计算雷达帧扫描开始和结束时间戳之间的imu相对位姿变换)
        if (!deskewInfo())
            return;

        // 对点云进行偏斜矫正，并投影到深度图中
        projectPointCloud();

        // 确定每根线的起始和结束点索引，并提取出偏斜矫正后点云及对应的点云信息
        cloudExtraction();

        // 发布有效点云和激光点云信息(包括每根线的起始和结束点索引、点深度、列索引)
        publishClouds();

        // 重置中间变量
        resetParameters();
    }

    // 转换雷达点云信息为可处理点云数据并缓存
    bool cachePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& laserCloudMsg)
    {
        // cache point cloud 缓存雷达点云消息
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;

        // convert cloud 将雷达点云消息转换成pcl点云数据结构
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();
        if (sensor == SensorType::VELODYNE || sensor == SensorType::LIVOX)
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);  
        }
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else if (sensor == SensorType::HESAI) {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpPandarCloudIn);
            laserCloudIn->points.resize(tmpPandarCloudIn->size());
            laserCloudIn->is_dense = tmpPandarCloudIn->is_dense;
            double time_begin = tmpPandarCloudIn->points[0].timestamp;
            for (size_t i = 0; i < tmpPandarCloudIn->size(); i++) {
                auto &src = tmpPandarCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.y * -1;
                dst.y = src.x;
                // dst.x = src.x;
                // dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;                    // 激光点在竖直方向上所属的线束序号
                //dst.tiSme = src.t * 1e-9f;
                dst.time = src.timestamp - time_begin;  // 当前激光点相对于当前激光帧第一个激光点的扫描时间， 单位秒s
            }
        }
        else
        {
            RCLCPP_ERROR_STREAM(get_logger(), "Unknown sensor type: " << int(sensor));
            rclcpp::shutdown();
        }

        // get timestamp 获取当前帧的时间戳
        cloudHeader = currentCloudMsg.header;
        timeScanCur = stamp2Sec(cloudHeader.stamp);
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time; // 计算当前帧扫描结束时时间戳
    
        if (debugLidarTimestamp) {
            // 这里输出的点云time时间戳单位是秒，值在0.0-0.1s之间，因为机械激光雷达扫描一圈得到一个点云的时间一般是0.1s
            std::cout << std::fixed << std::setprecision(12) 
                << "Lidar points: size="<< laserCloudIn->points.size()
                << " packet="<< laserCloudIn->points.size()
                << " start="<< laserCloudIn->points[0].time
                << " end=" << laserCloudIn->points.back().time << "s" << std::endl;
        }

        // remove Nan
        vector<int> indices;
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);

        // check dense flag 判断点云是否是稠密的(判断点云是否包含nan或者inf值)
        if (laserCloudIn->is_dense == false)
        {
            RCLCPP_ERROR(get_logger(), "Point cloud is not in dense format, please remove NaN points first!");
            rclcpp::shutdown();
        }

        // check ring channel 当雷达点云信息内包含时间戳字段，则设置去畸变标志位为true
        // we will skip the ring check in case of velodyne - as we calculate the ring value downstream (line 572)
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                if (sensor == SensorType::VELODYNE) {
                    ringFlag = 2;
                } else {
                    RCLCPP_ERROR(get_logger(), "Point cloud ring channel not available, please configure your point cloud data!");
                    rclcpp::shutdown();
                }
            }
        }

        // check point time
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                RCLCPP_WARN(get_logger(), "Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    // 点云偏斜矫正所需的imu数据的预处理(计算雷达帧扫描开始和结束时间戳之间的imu相对位姿变换)
    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan 当没有imu信息时，无法进行偏斜矫正
        if (imuQueue.empty() ||
            stamp2Sec(imuQueue.front().header.stamp) > timeScanCur ||
            stamp2Sec(imuQueue.back().header.stamp) < timeScanEnd)
        {
            RCLCPP_INFO(get_logger(), "Waiting for IMU data ...");
            return false;
        }

        // 通过积分获取当前雷达帧扫描开始和结束时间戳内的imu姿态信息(主要为获取imu帧相对姿态信息)
        imuDeskewInfo();

        // 获得imu里程计在当前雷达帧扫描开始和结束时间戳内的起始和结束帧，并计算两者之间的位姿变换(主要为获取imu帧相对位置信息)
        odomDeskewInfo();

        return true;
    }

    // 通过积分获取当前雷达帧扫描开始和结束时间戳内的imu姿态信息(主要为获取imu帧相对姿态信息)
    void imuDeskewInfo()
    {
        cloudInfo.imu_available = false;

        while (!imuQueue.empty())
        {
            // 丢弃早于当前雷达帧开始扫描时间戳的缓存的imu帧
            if (stamp2Sec(imuQueue.front().header.stamp) < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        // 如果没有imu数据，直接返回
        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            // 获取imu帧时间戳
            sensor_msgs::msg::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = stamp2Sec(thisImuMsg.header.stamp);

            // get roll, pitch, and yaw estimation for this scan 将用四元素表示的imu姿态信息转换成欧拉角表示
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imu_roll_init, &cloudInfo.imu_pitch_init, &cloudInfo.imu_yaw_init);
            
            // 如果imu帧时间戳大于雷达帧扫描结束的时间戳，则退出
            if (currentImuTime > timeScanEnd + 0.01)
                break;

            // 初始化第一帧imu姿态信息为0
            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity 获取imu角速度信息
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation 积分imu的姿态信息
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imu_available = true;
    }

    // 获得imu里程计在当前雷达帧扫描开始和结束时间戳内的起始和结束帧，并计算两者之间的位姿变换(主要为获取imu帧相对位置信息)
    void odomDeskewInfo()
    {
        cloudInfo.odom_available = false;

        // 丢弃早于当前雷达帧开始扫描时间的imu里程计帧
        while (!odomQueue.empty())
        {
            if (stamp2Sec(odomQueue.front().header.stamp) < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        // 如果没有imu里程计帧直接返回
        if (odomQueue.empty())
            return;

        if (stamp2Sec(odomQueue.front().header.stamp) > timeScanCur)
            return;

        // get start odometry at the beinning of the scan 获得imu里程计起始帧(也就是在雷达帧扫描开始和结束时间戳之间的第一个imu里程计帧)
        nav_msgs::msg::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (stamp2Sec(startOdomMsg.header.stamp) < timeScanCur)
                continue;
            else
                break;
        }

        // 将imu里程计帧的姿态信息转换为欧拉角表示
        tf2::Quaternion orientation;
        tf2::fromMsg(startOdomMsg.pose.pose.orientation, orientation);
        double roll, pitch, yaw;
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization 将imu里程计的位姿记录，用于将被发布出去地图优化的初始值
        cloudInfo.initial_guess_x = startOdomMsg.pose.pose.position.x;
        cloudInfo.initial_guess_y = startOdomMsg.pose.pose.position.y;
        cloudInfo.initial_guess_z = startOdomMsg.pose.pose.position.z;
        cloudInfo.initial_guess_roll = roll;
        cloudInfo.initial_guess_pitch = pitch;
        cloudInfo.initial_guess_yaw = yaw;

        cloudInfo.odom_available = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;

        if (stamp2Sec(odomQueue.back().header.stamp) < timeScanEnd)
            return;

        // 获得imu里程计结束帧(也就是在雷达帧扫描开始和结束时间戳之间的最后一个imu里程计帧)
        nav_msgs::msg::Odometry endOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (stamp2Sec(endOdomMsg.header.stamp) < timeScanEnd)
                continue;
            else
                break;
        }

        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        // 获得imu里程计起始帧位姿
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        // 获得imu里程计结束帧位姿
        tf2::fromMsg(endOdomMsg.pose.pose.orientation, orientation);
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);
        // 获得起始和结束帧之间的相对转换
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        // 获得起始和结束帧之间的位姿增量，其中姿态用欧拉角形式表示
        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }

    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        // 查找第一个时间戳大于等于当前点的imu数据指针
        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        // 如果不存在时间戳上大于当前点的imu数据帧，则直接返回最近的imu数据帧姿态信息
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else { // 如果存在时间戳大于当前点的imu数据帧，则利用该帧前一帧和该帧进行插值，获取当前点时间戳对应的姿态信息
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    // 如果传感器相对移动比较缓慢，则该移动对偏斜矫正影响较小，所以此处作者直接将移动量置为0，影响不大
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanEnd - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    // 对当前点进行偏斜矫正
    PointType deskewPoint(PointType *point, double relTime)
    {
        // 等待imu数据就绪才能进行处理
        if (deskewFlag == -1 || cloudInfo.imu_available == false)
            return *point;

        double pointTime = timeScanCur + relTime; // 当前点采集时的时间戳

        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur); // 获得当前点时间戳对应的imu姿态变化

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur); // 获得当前点时间戳对应imu位置变换(由于移动较慢，此处直接置为0)

        // 获取第一个点时间戳对应的位姿变化
        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start
        //获得当前点对应的imu位姿和第一个点对应的imu位姿之间的相对变换
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        // 变换当前点到第一个点所在的坐标系也即雷达坐标系(至此完成偏斜矫正)
        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    // 对点云进行偏斜矫正，并投影到深度图中
    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            float range = pointDistance(thisPoint);             // 计算当前点深度
            if (range < lidarMinRange || range > lidarMaxRange) // 如果深度在给定范围之外，则直接丢弃
                continue;

            int rowIdn = laserCloudIn->points[i].ring; // 获得当前点所在的线索引，也即在深度图中的行索引
            // if sensor is a velodyne (ringFlag = 2) calculate rowIdn based on number of scans
            if (ringFlag == 2) { 
                float verticalAngle =
                    atan2(thisPoint.z,
                        sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) *
                    180 / M_PI;
                rowIdn = (verticalAngle + (N_SCAN - 1)) / 2.0;
            }

            if (rowIdn < 0 || rowIdn >= N_SCAN)        // 如果在给定的线数范围之外，则直接丢弃
                continue;

            if (rowIdn % downsampleRate != 0)
                continue;

            int columnIdn = -1;
            if (sensor == SensorType::VELODYNE || sensor == SensorType::OUSTER || sensor == SensorType::HESAI)
            {
                float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;  // 计算当前点的水平倾角
                static float ang_res_x = 360.0/float(Horizon_SCAN);                 // 计算水平角分辨率
                columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2; // 计算当前点在在深度图中的列索引
                if (columnIdn >= Horizon_SCAN)  // 做一些冗余判断，防止越界
                    columnIdn -= Horizon_SCAN;
            }
            else if (sensor == SensorType::LIVOX)
            {
                columnIdn = columnIdnCountVec[rowIdn];
                columnIdnCountVec[rowIdn] += 1;
            }


            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            // 对当前点进行偏斜矫正
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

            // 填充深度图
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            // 记录完整点云
            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }

    // 确定每根线的起始和结束点索引，并提取出偏斜矫正后点云及对应的点云信息
    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry 确定每根线的起始和结束点索引，并提取出偏斜矫正后点云
        for (int i = 0; i < N_SCAN; ++i)
        {
            // 每根线上点云信息的起始索引未从起始点开始，原因是该部分提取出来的点云是用于特征提取的，起始点附近的点都无法有效计算曲率
            cloudInfo.start_ring_index[i] = count - 1 + 5;
            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.point_col_ind[count] = j;
                    // save range info
                    cloudInfo.point_range[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.end_ring_index[i] = count -1 - 5;
        }
    }
    
    // 发布有效点云和激光点云信息(包括每根线的起始和结束点索引、点深度、列索引)
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        // 发布提取出的有效点云
        cloudInfo.cloud_deskewed  = publishCloud(pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        // 发布激光点云信息(包括每根线的起始和结束点索引、点深度、列索引)
        pubLaserCloudInfo->publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;

    auto IP = std::make_shared<ImageProjection>(options);
    exec.add_node(IP);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Image Projection Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();
    return 0;
}
