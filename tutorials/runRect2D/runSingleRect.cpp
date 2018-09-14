/*
 * Copyright (c) 2015-2016, Graphics Lab, Georgia Tech Research Corporation
 * Copyright (c) 2015-2016, Humanoid Lab, Georgia Tech Research Corporation
 * Copyright (c) 2016, Personal Robotics Lab, Carnegie Mellon University
 * All rights reserved.
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

//C1:
//vel[2] = dart::math::random(-50,50);
//vel[3] = dart::math::random(-10,10);
//vel[4] = dart::math::random(-7,3);

//C2:
//vel[2] = dart::math::random(-30,30);
//vel[3] = dart::math::random(-5,5);
//vel[4] = dart::math::random(-7,7);

//R1:
//vel[2] = dart::math::random(-50,50);
//vel[3] = dart::math::random(-10,10);
//vel[4] = dart::math::random(-7,3);

//R2:
//vel[2] = dart::math::random(-30,30);
//vel[3] = dart::math::random(-5,5);
//vel[4] = dart::math::random(-7,7);


#include "dart/dart.hpp"
#include "dart/gui/gui.hpp"
#include "dart/collision/bullet/bullet.hpp"
#include "dart/collision/ode/ode.hpp"
#include <dart/utils/utils.hpp>
#include <math.h>

#include "tiny_dnn/tiny_dnn.h"

#include "dart/dart.hpp"
#include "dart/gui/gui.hpp"
#include "dart/collision/bullet/bullet.hpp"
#include "dart/collision/ode/ode.hpp"
#include <dart/utils/utils.hpp>
#include <math.h>

#include "tiny_dnn/tiny_dnn.h"

using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart;
using namespace utils;

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;
using namespace std;
#define NUMSAM 100

double GivenStartPos[] =
{0, 0, -2.31512, 0, 0.828924, 0,
    0, 0, -1.76583, 0, 0.332931, 0,
    0, 0, 2.73126, 0, 0.568451, 0,
    0, 0, -2.92437, 0, 0.337423, 0,
    0, 0, -3.09322, 0, 0.568391, 0,
    0, 0, 1.17353, 0, 0.712284, 0,
    0, 0, 0.169198, 0, 0.364375, 0,
    0, 0, 1.26412, 0, 0.937225, 0,
    0, 0, -2.84336, 0, 0.815257, 0,
    0, 0, 1.61107, 0, 0.993726, 0,
    0, 0, 3.03195, 0, 0.805862, 0,
    0, 0, -2.68489, 0, 0.742144, 0,
    0, 0, -0.399539, 0, 0.836546, 0,
    0, 0, -1.4143, 0, 0.551485, 0,
    0, 0, 2.49855, 0, 0.936446, 0,
    0, 0, 0.0284182, 0, 0.661404, 0,
    0, 0, -0.0378456, 0, 0.486301, 0,
    0, 0, -2.67821, 0, 0.650495, 0,
    0, 0, 2.60009, 0, 0.670823, 0,
    0, 0, -2.82691, 0, 0.83306, 0,
    0, 0, -2.3539, 0, 0.311107, 0,
    0, 0, 0.813945, 0, 0.815357, 0,
    0, 0, 2.44147, 0, 0.463236, 0,
    0, 0, 0.0834011, 0, 0.71378, 0,
    0, 0, 2.14577, 0, 0.488522, 0,
    0, 0, -0.201581, 0, 0.501049, 0,
    0, 0, 0.45022, 0, 0.861684, 0,
    0, 0, -0.0095497, 0, 0.968753, 0,
    0, 0, 2.45508, 0, 0.737395, 0,
    0, 0, -1.80484, 0, 0.800297, 0,
    0, 0, -1.4163, 0, 0.3021, 0,
    0, 0, 1.31834, 0, 0.956528, 0,
    0, 0, -1.14643, 0, 0.920893, 0,
    0, 0, 1.13943, 0, 0.57007, 0,
    0, 0, -2.21462, 0, 0.711031, 0,
    0, 0, 2.86142, 0, 0.689302, 0,
    0, 0, -0.573236, 0, 0.399274, 0,
    0, 0, -0.0721652, 0, 0.624821, 0,
    0, 0, -1.88648, 0, 0.523475, 0,
    0, 0, 0.950355, 0, 0.735144, 0,
    0, 0, -0.148083, 0, 0.57252, 0,
    0, 0, 2.52379, 0, 0.598548, 0,
    0, 0, -0.56352, 0, 0.391832, 0,
    0, 0, -2.12247, 0, 0.349744, 0,
    0, 0, -2.29268, 0, 0.848207, 0,
    0, 0, -0.299707, 0, 0.866261, 0,
    0, 0, -1.78915, 0, 0.775715, 0,
    0, 0, 2.26735, 0, 0.629884, 0,
    0, 0, 1.9953, 0, 0.82909, 0,
    0, 0, 0.834022, 0, 0.607531, 0,
    0, 0, 1.2705, 0, 0.991002, 0,
    0, 0, -1.32377, 0, 0.676198, 0,
    0, 0, -0.540175, 0, 0.703702, 0,
    0, 0, 1.44355, 0, 0.908485, 0,
    0, 0, 1.2977, 0, 0.819201, 0,
    0, 0, 0.157, 0, 0.624326, 0,
    0, 0, -0.0694722, 0, 0.767375, 0,
    0, 0, 2.61779, 0, 0.906118, 0,
    0, 0, -2.267, 0, 0.615243, 0,
    0, 0, -0.339145, 0, 0.521013, 0,
    0, 0, -0.378715, 0, 0.627272, 0,
    0, 0, -1.81258, 0, 0.999382, 0,
    0, 0, 0.731051, 0, 0.300417, 0,
    0, 0, 1.42839, 0, 0.523424, 0,
    0, 0, 1.1345, 0, 0.443683, 0,
    0, 0, 2.06533, 0, 0.366184, 0,
    0, 0, 0.814124, 0, 0.449697, 0,
    0, 0, -0.698544, 0, 0.966512, 0,
    0, 0, -1.45007, 0, 0.784518, 0,
    0, 0, 1.78358, 0, 0.595722, 0,
    0, 0, -3.07049, 0, 0.434277, 0,
    0, 0, 2.0089, 0, 0.395518, 0,
    0, 0, -2.03022, 0, 0.879848, 0,
    0, 0, -1.52575, 0, 0.463519, 0,
    0, 0, 0.846455, 0, 0.787205, 0,
    0, 0, 1.58927, 0, 0.768664, 0,
    0, 0, 0.617113, 0, 0.458905, 0,
    0, 0, -2.40371, 0, 0.8338, 0,
    0, 0, 0.552851, 0, 0.530766, 0,
    0, 0, -2.12568, 0, 0.639728, 0,
    0, 0, 0.35711, 0, 0.817298, 0,
    0, 0, 0.179374, 0, 0.517517, 0,
    0, 0, -0.434496, 0, 0.48119, 0,
    0, 0, -0.333851, 0, 0.633127, 0,
    0, 0, -2.64985, 0, 0.558819, 0,
    0, 0, 1.10733, 0, 0.659755, 0,
    0, 0, 2.79446, 0, 0.622488, 0,
    0, 0, -0.248599, 0, 0.662002, 0,
    0, 0, 0.663754, 0, 0.990552, 0,
    0, 0, -0.981323, 0, 0.679197, 0,
    0, 0, -3.04766, 0, 0.491947, 0,
    0, 0, 2.62541, 0, 0.486629, 0,
    0, 0, 2.16126, 0, 0.785817, 0,
    0, 0, -3.00284, 0, 0.412524, 0,
    0, 0, -0.398112, 0, 0.707695, 0,
    0, 0, 1.22949, 0, 0.495658, 0,
    0, 0, -3.11486, 0, 0.646984, 0,
    0, 0, 2.53402, 0, 0.848094, 0,
    0, 0, -3.10175, 0, 0.709351, 0,
    0, 0, 0.276603, 0, 0.572812, 0};

double GivenStartVel[] =
{0, 0, -0.826997, 0.262138, 0, 0,
    0, 0, 3.57729, 1.43437, 0, 0,
    0, 0, 0.388327, 2.64772, 0, 0,
    0, 0, 0.594004, 1.3692, 0, 0,
    0, 0, -8.66316, -0.660112, 0, 0,
    0, 0, 8.60873, 2.76934, 0, 0,
    0, 0, 3.07838, -0.672005, 0, 0,
    0, 0, 5.24396, -1.90038, 0, 0,
    0, 0, -3.43532, 1.06111, 0, 0,
    0, 0, -2.69323, -2.02369, 0, 0,
    0, 0, 5.06712, 1.21215, 0, 0,
    0, 0, 7.69414, -1.81832, 0, 0,
    0, 0, -0.445365, -2.0978, 0, 0,
    0, 0, -6.66986, -0.107861, 0, 0,
    0, 0, -8.78871, 3.23722, 0, 0,
    0, 0, -3.61934, 3.89314, 0, 0,
    0, 0, -8.18534, 3.58211, 0, 0,
    0, 0, -2.31716, -1.78335, 0, 0,
    0, 0, -0.711084, 3.52784, 0, 0,
    0, 0, 5.40409, 2.62254, 0, 0,
    0, 0, 3.76911, 2.94598, 0, 0,
    0, 0, 4.50824, 3.99566, 0, 0,
    0, 0, -3.87356, -1.19188, 0, 0,
    0, 0, 6.91963, -0.703354, 0, 0,
    0, 0, -1.69211, 0.298432, 0, 0,
    0, 0, -6.43345, -2.77024, 0, 0,
    0, 0, -9.33892, 0.275599, 0, 0,
    0, 0, 4.96585, 0.436671, 0, 0,
    0, 0, 6.84079, -2.72186, 0, 0,
    0, 0, -7.39145, -3.27208, 0, 0,
    0, 0, -1.71413, -3.78499, 0, 0,
    0, 0, -5.20178, -2.55283, 0, 0,
    0, 0, 3.04117, -2.79732, 0, 0,
    0, 0, -2.24549, -0.00207174, 0, 0,
    0, 0, 6.91151, 0.720869, 0, 0,
    0, 0, -7.03697, 3.86644, 0, 0,
    0, 0, 1.29797, -1.98299, 0, 0,
    0, 0, 9.2219, -2.99175, 0, 0,
    0, 0, 2.58538, -2.9863, 0, 0,
    0, 0, 6.06146, -2.01727, 0, 0,
    0, 0, -5.93499, -3.773, 0, 0,
    0, 0, -7.15958, 3.57989, 0, 0,
    0, 0, 7.71297, -3.26261, 0, 0,
    0, 0, -2.69322, -1.97554, 0, 0,
    0, 0, -0.893854, -1.20381, 0, 0,
    0, 0, 8.63349, 1.21317, 0, 0,
    0, 0, 8.17844, -1.999, 0, 0,
    0, 0, 0.119117, 0.80315, 0, 0,
    0, 0, -0.7551, 3.61094, 0, 0,
    0, 0, 6.49395, 1.51185, 0, 0,
    0, 0, 9.0883, 2.81016, 0, 0,
    0, 0, 0.288693, -3.17253, 0, 0,
    0, 0, 7.53131, -0.479691, 0, 0,
    0, 0, 4.31285, 2.40576, 0, 0,
    0, 0, -9.61815, 3.08825, 0, 0,
    0, 0, -8.69612, 1.70738, 0, 0,
    0, 0, 3.64098, -2.40356, 0, 0,
    0, 0, 7.80037, 0.351586, 0, 0,
    0, 0, 9.78725, -2.27574, 0, 0,
    0, 0, 0.293189, 3.05203, 0, 0,
    0, 0, 6.133, -1.07874, 0, 0,
    0, 0, -6.92791, 1.04391, 0, 0,
    0, 0, -9.98242, 2.18682, 0, 0,
    0, 0, -1.64553, 1.45995, 0, 0,
    0, 0, 6.7284, 1.67136, 0, 0,
    0, 0, -8.36525, 2.11204, 0, 0,
    0, 0, -5.72906, -3.35151, 0, 0,
    0, 0, 8.9509, -0.881171, 0, 0,
    0, 0, -4.3193, 2.21493, 0, 0,
    0, 0, -4.35688, -2.44826, 0, 0,
    0, 0, 9.66472, -2.04757, 0, 0,
    0, 0, -2.03713, 0.808081, 0, 0,
    0, 0, -6.84538, 3.9035, 0, 0,
    0, 0, -7.96725, -2.24471, 0, 0,
    0, 0, 5.8954, 1.56994, 0, 0,
    0, 0, 2.6686, -3.54847, 0, 0,
    0, 0, -3.62444, 1.59868, 0, 0,
    0, 0, 0.522466, 0.431291, 0, 0,
    0, 0, 4.05979, -2.85564, 0, 0,
    0, 0, 7.20451, 2.50613, 0, 0,
    0, 0, -3.67985, -2.91734, 0, 0,
    0, 0, 1.76238, 0.144669, 0, 0,
    0, 0, -2.59547, -0.855859, 0, 0,
    0, 0, -2.24338, -1.76566, 0, 0,
    0, 0, -4.92155, 1.37427, 0, 0,
    0, 0, 4.57217, 1.76614, 0, 0,
    0, 0, 8.80326, -1.42752, 0, 0,
    0, 0, 3.22711, -0.785337, 0, 0,
    0, 0, -6.99212, 1.36079, 0, 0,
    0, 0, 0.456154, 2.63391, 0, 0,
    0, 0, 2.85958, 0.369093, 0, 0,
    0, 0, 9.40173, -2.02613, 0, 0,
    0, 0, -0.884968, 2.53681, 0, 0,
    0, 0, 4.139, 1.66261, 0, 0,
    0, 0, 5.0342, 3.93226, 0, 0,
    0, 0, 5.07725, -2.63057, 0, 0,
    0, 0, -8.40166, -2.69535, 0, 0,
    0, 0, 4.93358, -0.508595, 0, 0,
    0, 0, 0.369568, -3.4652, 0, 0,
    0, 0, -5.68348, -0.913845, 0, 0};

#define UNSYMCONE false

class MyWindow : public dart::gui::SimWindow
{
public:
    
    MyWindow(const WorldPtr& world)
    {
        std::cout << "use DART/Ours/PDD?:" << std::endl;
        std::cin >> conModel;
        
        if (conModel == 1)
        {
            // Yifeng: OURS
            if (UNSYMCONE)
            {
                c1net.load(DART_DATA_PATH"/NN-contact-force/neuralnets/old/net-c1-unsym-feb26");
                c2net.load(DART_DATA_PATH"/NN-contact-force/neuralnets/old/net-c2-unsym-feb26");
                r1net.load(DART_DATA_PATH"/NN-contact-force/neuralnets/old/net-r1-unsym-feb26");
                r2net.load(DART_DATA_PATH"/NN-contact-force/neuralnets/old/net-r2-unsym-feb26");
            }
            else
            {
                c1net.load(DART_DATA_PATH"/NN-contact-force/neuralnets/old/net-c1-feb24");
                c2net.load(DART_DATA_PATH"/NN-contact-force/neuralnets/old/net-c2-feb25");
                r1net.load(DART_DATA_PATH"/NN-contact-force/neuralnets/old/net-r1-feb26");
                r2net.load(DART_DATA_PATH"/NN-contact-force/neuralnets/old/net-r2-feb25");
            }
        }
        else if (conModel == 2)
        {
            // PDD
            r1net.load(DART_DATA_PATH"/NN-contact-force/neuralnets/old/net-pdd-r1-feb24-2pi");
            r2net.load(DART_DATA_PATH"/NN-contact-force/neuralnets/old/net-pdd-r2-feb24-2pi");
        }
        
        setWorld(world);
        ts = 0;
        sampleCount = 0;
        hand_bd = mWorld->getSkeleton("hand skeleton")->getBodyNode(0);
        ground_bd = mWorld->getSkeleton("ground skeleton")->getBodyNode(0);
        
        gravity = mWorld->getGravity();
        
        // odometry to record the total rotation travelled
        theta_odo = mWorld->getSkeleton("hand skeleton")->getPosition(2);
    }
    
    void keyboard(unsigned char key, int x, int y) override
    {
        switch(key)
        {
            default:
                SimWindow::keyboard(key, x, y);
        }
    }
    
    void drawWorld() const override
    {
        // Make sure lighting is turned on and that polygons get filled in
        glEnable(GL_LIGHTING);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        
        SimWindow::drawWorld();
    }
    
    void timeStepping() override
    // substituting original DART contact solve with ours, keeping other steps in world->step() unchanged.
    {
        if(ts % 1000 == 0 && sampleCount < NUMSAM)
        {
            if (ts > 0)
            {
                Eigen::Vector6d EndPos = hand_bd->getSkeleton()->getPositions();
                EndLinearPos.push_back(Eigen::Vector3d(EndPos[3], EndPos[4], EndPos[5]));
                Eigen::Vector3d EndOriExp = Eigen::Vector3d(EndPos[0], EndPos[1], EndPos[2]);
                Eigen::Quaterniond EndOriQuat = dart::math::expToQuat(EndOriExp);
                EndAngularPos.push_back(EndOriQuat);
                sampleCount ++;
                firstContact = true;
            }
            if (conModel == 0)//GS data set. Generate new pos and vel
            {
                //Set positions
                auto Pi = dart::math::constants<double>::pi();
                Eigen::Vector6d pos = Eigen::Vector6d::Zero();
                pos[0] = 0.0; pos[1] = 0.0; pos[2] = dart::math::random(-1*Pi, Pi);
                pos[3] = 0.0; pos[5] = 0.0; pos[4] = dart::math::random(0.3, 1.0);
                if (sampleCount < NUMSAM){StartPos.push_back(pos);}
                mWorld->getSkeleton("hand skeleton")->getJoint(0)->setPositions(pos);
                
                // Set velocities
                Eigen::Vector6d vel = Eigen::Vector6d::Zero();
                vel[0] = 0.0; vel[1] = 0.0; vel[2] = dart::math::random(-10,10);
                vel[3] = dart::math::random(-4,4); //x vel
                vel[4] = 0.0; vel[5] = 0.0;
                if (sampleCount < NUMSAM){StartVel.push_back(vel);}
                // Create reference frames for setting the initial velocity
                Eigen::Isometry3d centerTf(Eigen::Isometry3d::Identity());
                centerTf.translation() = hand_bd->getSkeleton()->getCOM();
                SimpleFrame center(Frame::World(), "center", centerTf);
                Eigen::Vector3d v = Eigen::Vector3d(vel[3], vel[4], vel[5]);
                Eigen::Vector3d w = Eigen::Vector3d(vel[0], vel[1], vel[2]);
                center.setClassicDerivatives(v, w);
                SimpleFrame ref(&center, "root_reference");
                ref.setRelativeTransform(hand_bd->getTransform(&center));
                hand_bd->getSkeleton()->getJoint(0)->setVelocities(ref.getSpatialVelocity());
            }
            else {
                // Initialize position
                int curr_index = sampleCount * 6;
                double pos_0=GivenStartPos[curr_index]; double pos_1=GivenStartPos[curr_index+1];
                double pos_2=GivenStartPos[curr_index+2]; double pos_3=GivenStartPos[curr_index+3];
                double pos_4=GivenStartPos[curr_index+4]; double pos_5=GivenStartPos[curr_index+5];
                Eigen::Vector6d InputPos; InputPos<<pos_0,pos_1,pos_2,pos_3,pos_4,pos_5;
                mWorld->getSkeleton("hand skeleton")->getJoint(0)->setPositions(InputPos);
                
                // Initialize velocity
                double vel_0=GivenStartVel[curr_index]; double vel_1=GivenStartVel[curr_index+1];
                double vel_2=GivenStartVel[curr_index+2]; double vel_3=GivenStartVel[curr_index+3];
                double vel_4=GivenStartVel[curr_index+4]; double vel_5=GivenStartVel[curr_index+5];
                // Create reference frames for setting the initial velocity
                Eigen::Isometry3d centerTf(Eigen::Isometry3d::Identity());
                centerTf.translation() = hand_bd->getSkeleton()->getCOM();
                SimpleFrame center(Frame::World(), "center", centerTf);
                Eigen::Vector3d v = Eigen::Vector3d(vel_3, vel_4, vel_5);
                Eigen::Vector3d w = Eigen::Vector3d(vel_0, vel_1, vel_2);
                center.setClassicDerivatives(v, w);
                SimpleFrame ref(&center, "root_reference");
                ref.setRelativeTransform(hand_bd->getTransform(&center));
                hand_bd->getSkeleton()->getJoint(0)->setVelocities(ref.getSpatialVelocity());
            }
        }
        else if (ts % 1000 == 0 && sampleCount == NUMSAM)
        {
            if (conModel == 0)
            {
                cout<<"Starting positions are: "<<endl;
                for(int i=0; i < StartPos.size(); i++)
                {
                    cout<<StartPos[i][0]<<", "<<StartPos[i][1]<<", "<<StartPos[i][2]<<", "<<
                    StartPos[i][3]<<", "<<StartPos[i][4]<<", "<<StartPos[i][5]<<", "<<endl;
                }
                cout<<" "<<endl;
                cout<<"Starting velocities are: "<<endl;
                for(int i=0; i < StartVel.size(); i++)
                {
                    cout<<StartVel[i][0]<<", "<<StartVel[i][1]<<", "<<StartVel[i][2]<<", "<<
                    StartVel[i][3]<<", "<<StartVel[i][4]<<", "<<StartVel[i][5]<<", "<<endl;
                }
            }
            cout<<" "<<endl;
            cout<<"Final linear positions are: "<<endl;
            for(int i=0; i < EndLinearPos.size(); i++)
            {
                cout<<EndLinearPos[i][0]<<", "<<EndLinearPos[i][1]<<", "<<EndLinearPos[i][2]<<", "<<endl;
            }
            cout<<" "<<endl;
            cout<<"All end orientation in quaternion in w x y z: "<<endl;
            for(int i=0; i < EndAngularPos.size(); i++)
            {
                Eigen::Quaterniond cur = EndAngularPos[i];
                cout<< cur.w()<<", "<<cur.x()<<", "<<cur.y()<<", "<<cur.z()<<", "<<endl;
            }
            cout<<" "<<endl;
            cout<<"All first contact impulses are: "<<endl;
            for(int i=0; i < FirstContactImpulse.size(); i++) {
                cout<<FirstContactImpulse[i][0]<<", "<<FirstContactImpulse[i][1]<<", "<<FirstContactImpulse[i][2]
                <<", " << FirstContactImpulse[i][3]<<", "<<FirstContactImpulse[i][4]<<", "
                <<FirstContactImpulse[i][5]<<", "<<endl;
            }
            sampleCount ++;
        }
        if (ts != 0)
        {
            auto a = hand_bd->getSpatialVelocity(Frame::World(),Frame::World());
            theta_odo = theta_odo + a[2] * mWorld->getTimeStep();
        }
        
        // Integrate velocity for unconstrained skeletons
        for (size_t i=0; i < mWorld->getNumSkeletons(); i++)
        {
            auto skel = mWorld->getSkeleton(i);
            if (!skel->isMobile())
                continue;
            
            skel->computeForwardDynamics();
            skel->integrateVelocities(mWorld->getTimeStep());
        }
        
        pos = hand_bd->getSkeleton()->getPositions();
        vel_uncons = hand_bd->getSpatialVelocity(Frame::World(),Frame::World());
        Eigen::VectorXd in_vec = Eigen::VectorXd::Zero(5);
        in_vec << sin(pos[2]), cos(pos[2]), vel_uncons[2], vel_uncons[3], vel_uncons[4];
        
        //        std::cout << "t=" << ts ;
//               std::cout << in_vec.transpose() << std::endl;
//                std::cout << pos[3] << " " << pos[4] << " " << theta_odo<< std::endl;
        
        auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
        auto collisionGroup = mWorld->getConstraintSolver()->getCollisionGroup();
        dart::collision::CollisionOption option;
        dart::collision::CollisionResult result;
        bool collision = collisionGroup->collide(option, &result);
        
        if (collision)
        {
            tempImp = Eigen::Vector6d::Zero();
            if (conModel == 0)
            {
                mWorld->getConstraintSolver()->solve();
                
            }
            if (conModel == 1)
            {
                myContactSolve(in_vec, result);
            }
            
            if (conModel == 2)
            {
                pddContactSolve(in_vec, result);
            }
            if(firstContact == true)
            {
                firstContact = false;
                Eigen::Vector6d firstImp = hand_bd->getConstraintImpulse() + tempImp;
                if (sampleCount < NUMSAM){FirstContactImpulse.push_back(firstImp);}
            }
        }
        
        // Compute velocity changes given constraint impulses
        for (size_t i=0; i < mWorld->getNumSkeletons(); i++)
        {
            auto skel = mWorld->getSkeleton(i);
            if (!skel->isMobile())
                continue;
            
            skel->computeImpulseForwardDynamics();
            
            skel->integratePositions(mWorld->getTimeStep());
            skel->clearInternalForces();
            skel->clearExternalForces();
            skel->resetCommands();
            skel->clearConstraintImpulses();
        }
        ts++;
    }
    
    void myContactSolve(Eigen::VectorXd in_vec, const dart::collision::CollisionResult& result)
    {
        Eigen::VectorXd in_c = in_vec;
        in_c[2] = in_c[2] / 10.0; // "normalize" input scale
        
        // TODO: a temporary fix utilizing linearity of the meta cone.
        // scale small inputs to larger region where NN has smaller relative error
        // otherwise artifects may be obvious
        Eigen::Vector3d velAbs;
        velAbs << in_c[2], in_c[3], in_c[4];
        velAbs = velAbs.cwiseAbs();
        if (velAbs.maxCoeff() < 0.1)
        {
            in_c[2] = in_c[2] * (2.5/velAbs.maxCoeff());
            in_c[3] = in_c[3] * (2.5/velAbs.maxCoeff());
            in_c[4] = in_c[4] * (2.5/velAbs.maxCoeff());
        }
        
        vec_t input_c;
        input_c.assign(in_c.data(), in_c.data()+5);
        
        label_t c1output = c1net.predict_label(input_c);
        label_t c2output = c2net.predict_label(input_c);
        
        if (result.getNumContacts() == 2)
        {
            if (velAbs[0] <= 5 && velAbs[1] <= 10 && in_c[4] > -7 && in_c[4] < 3)
            {
                if (c1output == 1)
                {
                    // set 2 ball constraints
//                    std::cout << "vola1" << std::endl;
                    
                    auto pos1 = result.getContact(0).point;
                    auto pos2 = result.getContact(1).point;
                    auto constraint1 = std::make_shared<dart::constraint::BallJointConstraint>(hand_bd, ground_bd, pos1);
                    auto constraint2 = std::make_shared<dart::constraint::BallJointConstraint>(hand_bd, ground_bd, pos2);
                    mWorld->getConstraintSolver()->addConstraint(constraint1);
                    mWorld->getConstraintSolver()->addConstraint(constraint2);
                    
                    // disable ground from contact detection
                    mWorld->getConstraintSolver()->getCollisionGroup()->
                    removeShapeFramesOf(ground_bd->getSkeleton().get());
                    
                    // solve for impulses needed to maintain ball joints
                    mWorld->getConstraintSolver()->solve();
                    
                    // restore
                    mWorld->getConstraintSolver()->removeConstraint(constraint1);
                    mWorld->getConstraintSolver()->removeConstraint(constraint2);
                    mWorld->getConstraintSolver()->getCollisionGroup()->
                    addShapeFramesOf(ground_bd->getSkeleton().get());
                }
                else if (c1output == 0)
                {
                    // run regreesor
                    auto pos1 = result.getContact(0).point;
                    auto pos2 = result.getContact(1).point;
                    
                    Eigen::VectorXd in_r = Eigen::VectorXd::Zero(5);
                    in_r = in_vec;
                    in_r[2] = in_r[2] / 10.0;
                    
                    // TODO: a temporary fix utilizing linearity of the meta cone.
                    Eigen::Vector3d velAbs;
                    velAbs << in_r[2], in_r[3], in_r[4];
                    velAbs = velAbs.cwiseAbs();
                    double scale = 1.0;
                    if (velAbs.maxCoeff() < 0.5 && std::abs(in_r[2] * (2.5/velAbs.maxCoeff()))<5.0)
                    {
                        in_r[2] = in_r[2] * (2.5/velAbs.maxCoeff());
                        in_r[3] = in_r[3] * (2.5/velAbs.maxCoeff());
                        in_r[4] = in_r[4] * (2.5/velAbs.maxCoeff());
                        scale = 2.5/velAbs.maxCoeff();
                    }
                    
                    vec_t input_r;
                    input_r.assign(in_r.data(), in_r.data()+5);
                    
                    vec_t r1output = r1net.predict(input_r);
                    // scaled down 100 times when training, should be okay if just train with large labels
                    double fric = *(r1output.begin()) * 100.0 / scale;
//                    std::cout << fric << std::endl;
                    Eigen::Vector3d friction;
                    friction << fric, 0.0, 0.0;
                    
                    // decouple instead of blindly run one step
                    hand_bd -> clearExternalForces();
                    mWorld->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
                    hand_bd -> addExtForce(friction, (pos1+pos2)/2.0, false, false);
                    
                    hand_bd -> addConstraintImpulse(hand_bd->getAspectState().mFext * mWorld->getTimeStep());
                    hand_bd -> clearExternalForces();
                    hand_bd -> getSkeleton()->computeImpulseForwardDynamics();
                    
                    tempImp = hand_bd->getConstraintImpulse();
                    
                    hand_bd -> setConstraintImpulse(Eigen::Vector6d::Zero());
                    hand_bd -> setFrictionCoeff(0.0);
                    ground_bd -> setFrictionCoeff(0.0);
                    mWorld->getConstraintSolver()->solve();

                    // restore
                    hand_bd -> setFrictionCoeff(1.0);
                    ground_bd ->setFrictionCoeff(1.0);
                    mWorld->setGravity(gravity);
                    
//                    std::cout << "vola0" << std::endl;
                }
                else // c1out == 2
                {
                    // do nothing, ignore collision solving
                    
                    mWorld->getConstraintSolver()->getCollisionGroup()->
                    removeShapeFramesOf(ground_bd->getSkeleton().get());
                    
                    mWorld->getConstraintSolver()->solve();
                    // restore
                    mWorld->getConstraintSolver()->getCollisionGroup()->
                    addShapeFramesOf(ground_bd->getSkeleton().get());
                    
//                    std::cout << "vola2" << std::endl;
                }
            }
            else
            {
                std::cout << "warning: c1 OOR " << in_vec.transpose() << std::endl;
                mWorld->getConstraintSolver()->solve();
            }
        }
        else if (result.getNumContacts() == 4)
        {
            if (velAbs[0] <= 3 && velAbs[1] <= 5 && velAbs[2] <= 7)
            {
                if (c2output == 1)
                {
                    // set 1 weld constraints
//                    std::cout << "alov1" << std::endl;
                    
                    auto constraint = std::make_shared<dart::constraint::WeldJointConstraint>(hand_bd, ground_bd);
                    
                    mWorld->getConstraintSolver()->addConstraint(constraint);
                    
                    // disable ground from contact detection
                    mWorld->getConstraintSolver()->getCollisionGroup()->
                    removeShapeFramesOf(ground_bd->getSkeleton().get());
                    
                    mWorld->getConstraintSolver()->solve();
                    
                    // restore
                    mWorld->getConstraintSolver()->removeConstraint(constraint);
                    mWorld->getConstraintSolver()->getCollisionGroup()->
                    addShapeFramesOf(ground_bd->getSkeleton().get());
                }
                else if (c2output == 0)
                {
                    // run regreesor
                    auto pos1 = result.getContact(0).point;
                    auto pos3 = result.getContact(2).point;
                    
                    Eigen::VectorXd in_r = Eigen::VectorXd::Zero(5);
                    in_r = in_vec;
                    in_r[2] = in_r[2] / 10.0;
                    
                    // TODO: a temporary fix utilizing linearity of the meta cone.
                    Eigen::Vector3d velAbs;
                    velAbs << in_r[2], in_r[3], in_r[4];
                    velAbs = velAbs.cwiseAbs();
                    double scale = 1.0;
                    if (velAbs.maxCoeff() < 0.5)
                    {
                        in_r[2] = in_r[2] * (2.5/velAbs.maxCoeff());
                        in_r[3] = in_r[3] * (2.5/velAbs.maxCoeff());
                        in_r[4] = in_r[4] * (2.5/velAbs.maxCoeff());
                        scale = 2.5/velAbs.maxCoeff();
                    }
                    
                    vec_t input_r;
                    input_r.assign(in_r.data(), in_r.data()+5);
                    
                    vec_t r2output = r2net.predict(input_r);
                    // scaled down 100 times when training
                    double fric = *(r2output.begin()) * 100.0 / scale;
                    
//                    std::cout << fric << std::endl;
                    Eigen::Vector3d friction;
                    friction << fric, 0.0, 0.0;

                    
                    // decouple instead of blindly run one step
                    hand_bd -> clearExternalForces();
                    mWorld->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
                    hand_bd -> addExtForce(friction, (pos1+pos3)/2.0, false, false);
                    
                    hand_bd -> addConstraintImpulse(hand_bd->getAspectState().mFext * mWorld->getTimeStep());
                    hand_bd -> clearExternalForces();
                    hand_bd -> getSkeleton()->computeImpulseForwardDynamics();
                    
                    tempImp = hand_bd->getConstraintImpulse();
                    
                    hand_bd -> setConstraintImpulse(Eigen::Vector6d::Zero());
                    hand_bd -> setFrictionCoeff(0.0);
                    ground_bd -> setFrictionCoeff(0.0);
                    mWorld->getConstraintSolver()->solve();

                    // restore
                    hand_bd -> setFrictionCoeff(1.0);
                    ground_bd ->setFrictionCoeff(1.0);
                    mWorld->setGravity(gravity);
                    
//                    std::cout << "alov0" << std::endl;
                }
                else // c2out == 2
                {
                    // do nothing, ignore collision solving
                    
                    mWorld->getConstraintSolver()->getCollisionGroup()->
                    removeShapeFramesOf(ground_bd->getSkeleton().get());
                    
                    mWorld->getConstraintSolver()->solve();
                    // restore
                    mWorld->getConstraintSolver()->getCollisionGroup()->
                    addShapeFramesOf(ground_bd->getSkeleton().get());
                    
//                    std::cout << "alov2" << std::endl;
                }
            }
            else
            {
                std::cout << "warning: c2 OOR " << in_vec.transpose() << std::endl;
                mWorld->getConstraintSolver()->solve();
            }
        }
    }
    
    void pddContactSolve(Eigen::VectorXd in_vec, const dart::collision::CollisionResult& result)
    {
        in_vec[2] = in_vec[2] / 10.0;
        
        if (result.getNumContacts() == 2)
        {
            if (std::abs(in_vec[2]) <= 5 && std::abs(in_vec[3]) <= 10 && in_vec[4] > -7 && in_vec[4] < 3)
            {
                // run regreesor
                auto pos1 = result.getContact(0).point;
                auto pos2 = result.getContact(1).point;
                
                // TODO: a temporary fix utilizing linearity of the meta cone.
                Eigen::Vector3d velAbs;
                velAbs << in_vec[2], in_vec[3], in_vec[4];
                velAbs = velAbs.cwiseAbs();
                double scale = 1.0;
                if (velAbs.maxCoeff() < 0.5 && std::abs(in_vec[2] * (2.5/velAbs.maxCoeff()))<5.0)
                {
                    in_vec[2] = in_vec[2] * (2.5/velAbs.maxCoeff());
                    in_vec[3] = in_vec[3] * (2.5/velAbs.maxCoeff());
                    in_vec[4] = in_vec[4] * (2.5/velAbs.maxCoeff());
                    scale = 2.5/velAbs.maxCoeff();
                }
                
                vec_t input_r;
                input_r.assign(in_vec.data(), in_vec.data()+5);
                
                vec_t r1output = r1net.predict(input_r);
                // scaled down 100 times when training
                double fric = *(r1output.begin()) * 100.0 / scale;
                double normal = *(r1output.begin()+1) * 100.0 / scale;
                Eigen::Vector3d friction;
                friction << fric, normal, 0.0;
                
                hand_bd -> clearExternalForces();
                mWorld->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
                hand_bd -> addExtForce(friction, (pos1+pos2)/2.0, false, false);
                
                hand_bd -> addConstraintImpulse(hand_bd->getAspectState().mFext * mWorld->getTimeStep());
                hand_bd -> clearExternalForces();
                
                mWorld->setGravity(gravity);
                
                //                std::cout << "vola0" << std::endl;
            }
            else
            {
                std::cout << "warning: r1 OOR " << in_vec.transpose() << std::endl;
                mWorld->getConstraintSolver()->solve();
            }
        }
        else if (result.getNumContacts() == 4)
        {
            if (std::abs(in_vec[2]) <= 3 && std::abs(in_vec[3]) <= 5 && std::abs(in_vec[4]) <= 7)
            {
                // run regreesor
                auto pos1 = result.getContact(0).point;
                auto pos3 = result.getContact(2).point;
                
                // TODO: a temporary fix utilizing linearity of the meta cone.
                Eigen::Vector3d velAbs;
                velAbs << in_vec[2], in_vec[3], in_vec[4];
                velAbs = velAbs.cwiseAbs();
                double scale = 1.0;
                if (velAbs.maxCoeff() < 0.5)
                {
                    in_vec[2] = in_vec[2] * (2.5/velAbs.maxCoeff());
                    in_vec[3] = in_vec[3] * (2.5/velAbs.maxCoeff());
                    in_vec[4] = in_vec[4] * (2.5/velAbs.maxCoeff());
                    scale = 2.5/velAbs.maxCoeff();
                }
                
                vec_t input_r;
                input_r.assign(in_vec.data(), in_vec.data()+5);
                
                vec_t r2output = r2net.predict(input_r);
                // scaled down 100 times when training
                double fric = *(r2output.begin()) * 100.0 / scale;
                double normal = *(r2output.begin()+1) * 100.0 / scale;
                double loc = *(r2output.begin()+2) / 100.0;
                
                //                    std::cout << fric << std::endl;
                Eigen::Vector3d friction;
                friction << fric, normal, 0.0;
                
                
                // decouple instead of blindly run one step
                hand_bd -> clearExternalForces();
                mWorld->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
                
                hand_bd -> addExtForce(friction, (pos1+pos3)/2.0 + Eigen::Vector3d(-loc,0,0), false, false);
                
                hand_bd -> addConstraintImpulse(hand_bd->getAspectState().mFext * mWorld->getTimeStep());
                hand_bd -> clearExternalForces();
                
                mWorld->setGravity(gravity);
                
                //                std::cout << "alov0" << std::endl;
            }
            else
            {
                std::cout << "warning: c2 OOR " << in_vec.transpose() << std::endl;
                mWorld->getConstraintSolver()->solve();
            }
        }
    }

    
    int ts;
    double theta_odo;
    
    Eigen::Vector6d vel_uncons;
    Eigen::Vector6d pos;
    network<sequential> c1net;
    network<sequential> c2net;
    network<sequential> r1net;
    network<sequential> r2net;
    
    Eigen::Vector3d gravity;
    
    BodyNodePtr hand_bd;
    BodyNodePtr ground_bd;
    
    int sampleCount;
    bool firstContact = true;
    int conModel;
    
    vector<Eigen::Vector6d> StartPos;
    vector<Eigen::Vector6d> StartVel;
    vector<Eigen::Vector3d> EndLinearPos;
    vector<Eigen::Quaterniond> EndAngularPos;
    vector<Eigen::Vector6d> FirstContactImpulse;
    Eigen::Vector6d tempImp;
    
protected:
};


int main(int argc, char* argv[])
{
    WorldPtr world = SkelParser::readWorld(DART_DATA_PATH"/NN-contact-force/skel/singleBody_test.skel");
    assert(world != nullptr);
    MyWindow window(world);
    std::cout << "space bar: simulation on/off" << std::endl;
    glutInit(&argc, argv);
    window.initWindow(640, 480, "Simple Test");
    glutMainLoop();
}
