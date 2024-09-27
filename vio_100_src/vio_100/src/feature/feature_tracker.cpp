

#include "feature_tracker.h"
#include "../utility/tic_toc.h"
#include <cmath>
#include<fstream>
bool FeatureTracker::inBorder(const cv::Point2f& pt) {
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return !(BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE);
}

double distance(cv::Point2f pt1, cv::Point2f pt2) {
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void reduceVector(vector<cv::Point2f>& v, vector<uchar> status) {
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int>& v, vector<uchar> status) {
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker() {
    stereo_cam = 0;
    n_id = 0;
    hasPrediction = false;
    sum_n = 0;
}

void FeatureTracker::setMask() {
    if (mask_global.cols != 0)mask = mask_global.clone();
    else mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>>& a, const pair<int, pair<cv::Point2f, int>>& b) {
        return a.first > b.first;
    });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto& it : cnt_pts_id) {
        if (mask.at<uchar>(it.second.first) == 255) {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);

        }
    }
}

void FeatureTracker::addPoints() {
    for (auto& p : n_pts) {
        cur_pts.push_back(p);
        ids.push_back(n_id++);
        track_cnt.push_back(1);
    }
}

double FeatureTracker::distance(cv::Point2f& pt1, cv::Point2f& pt2) {
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void FeatureTracker::AddImgsRaw(double _cur_time, const cv::Mat& _img) {
    img_raw[_cur_time] = _img;
}


map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTracker::trackImage(double _cur_time, const cv::Mat& _img, const cv::Mat& _img1) {
    TicToc t_r;
    cur_time = _cur_time;
    cur_img = _img;
    row = cur_img.rows;
    col = cur_img.cols;
    cv::Mat rightImg = _img1;
    cur_pts.clear();


    if (USE_GRAY_SCALE) {
        cur_img = _img.clone();
        if (exposure_time_file[0] != 0) {

            if (prev_time == 0) {
                std::string line;
                std::ifstream in = std::ifstream(exposure_time_file);
                std::cout << exposure_time_file << std::endl;
                if (in.fail())
                    cout << "File not found" << endl;
                getline(in, line);
                while (getline(in, line)  && in.good() ) {

                    vector<double>b;
                    std::stringstream sin(line);
                    string field;
                    string::size_type size;

                    while (getline(sin, field, ' ')) {
                        double d = stod(field, &size);
                        b.push_back(d);
                    }

                    uint64_t time = b[1] * 1e3;
                    std::cout << time << std::endl;
                    double exp = b[2];
                    exposure_times.push(make_pair(time, exp));
                }
            }
        }

        if (prev_time != 0) {
            double cur_exposure_time = -1;
            double prev_exposure_time = -1;
            if (exposure_time_file[0] == 0) {
                cur_exposure_time = cv::sum(cur_img)(0);
                prev_exposure_time = cv::sum(prev_img)(0);
            } else {
                uint64_t cur_time_i = cur_time * 1e3;
                uint64_t prev_time_i = prev_time * 1e3;
                while (exposure_times.front().first < prev_time_i)
                    exposure_times.pop();
                assert(exposure_times.front().first == prev_time_i);
                prev_exposure_time = exposure_times.front().second;
                exposure_times.pop();
                assert(exposure_times.front().first == cur_time_i);
                cur_exposure_time = exposure_times.front().second;
            }

            double scale = cur_exposure_time / prev_exposure_time;
            LOG_OUT << "灰度缩放:" << scale << "," << prev_exposure_time << "," << cur_exposure_time << std::endl;
            if (scale != 1) {
                if (scale > 1) {
                    for (int i = 0; i < prev_img.rows; ++i) {
                        for (int j = 0; j < prev_img.cols; ++j) {
                            int result = static_cast<int>(prev_img.at<uchar>(i, j) * scale);
                            prev_img.at<uchar>(i, j) = (result > 255) ? 255 : static_cast<uchar>(result);
                        }
                    }
                } else {
                    scale = 1 / scale;
                    int result;
                    for (int i = 0; i < cur_img.rows; ++i) {
                        for (int j = 0; j < cur_img.cols; ++j) {
                            result = static_cast<int>(cur_img.at<uchar>(i, j) * scale);
                            cur_img.at<uchar>(i, j) = (result > 255) ? 255 : static_cast<uchar>(result);
                        }
                    }
                }
            }
        }
    } else
        cur_img = _img;

    if (prev_pts.size() > 0) {
        vector<uchar> status;

        {
            TicToc t_o;

            vector<float> err;
            if (hasPrediction) {
                cur_pts = predict_pts;
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(OPTICAL_WINSIZE, OPTICAL_WINSIZE), OPTICAL_LAYER_NUM,
                                         cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

                int succ_num = 0;
                for (size_t i = 0; i < status.size(); i++) {
                    if (status[i])
                        succ_num++;
                }
                if (succ_num < 10)
                    cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(OPTICAL_WINSIZE, OPTICAL_WINSIZE), OPTICAL_LAYER_NUM);
            } else
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(OPTICAL_WINSIZE, OPTICAL_WINSIZE), OPTICAL_LAYER_NUM);


            // reverse check
            if (FLOW_BACK) {
                vector<uchar> reverse_status;
                vector<cv::Point2f> reverse_pts = prev_pts;
                cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(OPTICAL_WINSIZE, OPTICAL_WINSIZE), 1,
                                         cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                for (size_t i = 0; i < status.size(); i++) {
                    if (status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
                        status[i] = 1;
                    else
                        status[i] = 0;
                }


            }
        }


        for (int i = 0; i < int(cur_pts.size()); i++)
            if (status[i] && inBorder(cur_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
    }
    for (auto& n : track_cnt)
        n++;

    {
        rejectWithF();
        TicToc t_m;
        setMask();
        //extract MAX_CNT-100 number of features with large energy.
        int n_max_cnt = MAX_CNT - 100 - static_cast<int>(cur_pts.size());

        if (n_max_cnt > 0) {
            TicToc t_t;
            if (mask.empty())
                LOG_OUT << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                LOG_OUT << "mask type wrong " << endl;

            cv::goodFeaturesToTrack(cur_img, n_pts, MAX_CNT - 100 - cur_pts.size(), 0.01, MIN_DIST, mask);

        } else
            n_pts.clear();

        TicToc t_a;
        addPoints();

        //extract 100 number of features with small energy.
        if (mask_global.cols != 0)mask = mask_global.clone();
        else mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));
        for (auto& it : cur_pts)
            cv::circle(mask, it, 50, 0, -1);
        cv::goodFeaturesToTrack(cur_img, n_pts, 100, 0.001, 50, mask);
        addPoints();


    }

    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    if (!_img1.empty() && stereo_cam) {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        if (!cur_pts.empty()) {
            vector<cv::Point2f> reverseLeftPts;
            vector<uchar> status, statusRightLeft;

            {

                TicToc t_check;
                vector<float> err;
                // cur left ---- cur right
                cur_right_pts = cur_pts;
                cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts, status, err, cv::Size(OPTICAL_WINSIZE, OPTICAL_WINSIZE), OPTICAL_LAYER_NUM);
                // reverse check cur right ---- cur left
                if (FLOW_BACK) {
                    cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err, cv::Size(OPTICAL_WINSIZE, OPTICAL_WINSIZE), OPTICAL_LAYER_NUM);
                    for (size_t i = 0; i < status.size(); i++) {
                        if (status[i] && statusRightLeft[i] && !inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                            status[i] = 1;
                        else
                            status[i] = 0;
                    }
                }
            }

            ids_right = ids;
            reduceVector(cur_right_pts, status);
            reduceVector(ids_right, status);
            rejectWithFStereo();
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);

        }
        prev_un_right_pts_map = cur_un_right_pts_map;
    }
    if (SHOW_TRACK)
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

    prev_img = _img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;
    hasPrediction = false;

    prevLeftPtsMap.clear();
    for (size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (size_t i = 0; i < ids.size(); i++) {
        int feature_id = ids[i];
        double x, y, w;

        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        w = 1;

        double p_u, p_v;
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;
        int camera_id = 0;
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x;
        velocity_y = pts_velocity[i].y;



        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, w, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }

    if (!_img1.empty() && stereo_cam) {
        for (size_t i = 0; i < ids_right.size(); i++) {
            int feature_id = ids_right[i];
            double x, y, w;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            w = 1;
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
            int camera_id = 1;
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, w, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
        }
    }
    return featureFrame;
}

void FeatureTracker::rejectWithF() {
    if (cur_pts.size() >= 8) {
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++) {
            Eigen::Vector3d tmp_p;
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);


    }
}

void FeatureTracker::rejectWithFStereo() {
    if (cur_pts.size() >= 8) {

        TicToc t_f;
        vector<cv::Point2f> un_cur_pts, un_cur_right_pts;

        for (unsigned int i = 0; i < cur_right_pts.size(); i++) {
            Eigen::Vector3d tmp_p;

            m_camera[1]->liftProjective(Eigen::Vector2d(cur_right_pts[i].x, cur_right_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_right_pts.push_back( cv::Point2f(tmp_p.x(), tmp_p.y()));

            for (int j = 0; j < (int)ids.size(); j++) {
                if (ids[j] == ids_right[i]) {
                    m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[j].x, cur_pts[j].y), tmp_p);
                    tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
                    tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
                    un_cur_pts.push_back(cv::Point2f(tmp_p.x(), tmp_p.y()));
                }
            }
        }

        assert(un_cur_pts.size() == un_cur_right_pts.size());

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_cur_right_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);


        reduceVector(cur_right_pts, status);
        reduceVector(ids_right, status);

    }
}

void FeatureTracker::readIntrinsicParameter(const vector<string>& calib_file) {
    for (size_t i = 0; i < calib_file.size(); i++) {
        printf("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

void FeatureTracker::showUndistortion(const string& name) {
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++) {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
        }
    for (int i = 0; i < int(undistortedp.size()); i++) {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(2);
}

vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f>& pts, camodocal::CameraPtr cam) {
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++) {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int>& ids, vector<cv::Point2f>& pts,
                                                map<int, cv::Point2f>& cur_id_pts, map<int, cv::Point2f>& prev_id_pts) {
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
        cur_id_pts.insert(make_pair(ids[i], pts[i]));

    // caculate points velocity
    if (!prev_id_pts.empty()) {
        double dt = cur_time - prev_time;

        for (unsigned int i = 0; i < pts.size(); i++) {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end()) {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            } else
                pts_velocity.push_back(cv::Point2f(0, 0));

        }
    } else {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
            pts_velocity.push_back(cv::Point2f(0, 0));
    }
    return pts_velocity;
}

void FeatureTracker::drawTrack(const cv::Mat& imLeft, const cv::Mat& imRight,
                               vector<int>& curLeftIds,
                               vector<cv::Point2f>& curLeftPts,
                               vector<cv::Point2f>& curRightPts,
                               map<int, cv::Point2f>& prevLeftPtsMap) {
    int cols = imLeft.cols;
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, cv::COLOR_GRAY2BGR);

    for (size_t j = 0; j < curLeftPts.size(); j++) {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    if (!imRight.empty() && stereo_cam) {
        for (size_t i = 0; i < curRightPts.size(); i++) {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            for (size_t j = 0; j < ids.size(); j++) {
                if (ids_right[i] == ids[j]) {
                    cv::Point2f leftPt = curLeftPts[j];
                    leftPt.x += cols;
                    cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);

                }
            }
        }
    }

    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++) {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if (mapIt != prevLeftPtsMap.end())
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    }

    cv::imshow("tracking", imTrack);
    cv::waitKey(2);

}


void FeatureTracker::setPrediction(map<int, Eigen::Vector3d>& predictPts) {
    hasPrediction = true;
    predict_pts.clear();
    map<int, Eigen::Vector3d>::iterator itPredict;
    for (size_t i = 0; i < ids.size(); i++) {
        //printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        itPredict = predictPts.find(id);
        if (itPredict != predictPts.end()) {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            if (inBorder(cv::Point2f(tmp_uv.x(), tmp_uv.y())))
                predict_pts.push_back(prev_pts[i]);
            else
                predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));

        } else
            predict_pts.push_back(prev_pts[i]);
    }
}


void FeatureTracker::removeOutliers(set<int>& removePtsIds) {
    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++) {
        itSet = removePtsIds.find(ids[i]);
        if (itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}


cv::Mat FeatureTracker::getTrackImage() {
    return imTrack;
}
