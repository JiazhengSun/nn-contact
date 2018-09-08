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

#define UNSYMCONE false
#define NUMSAM 100

double StartPos[] = {-2.31512, 1.60602, -0.259809, 0, 1.09915, 0,
    -0.731978, 0.121997, 2.07952, 0, 0.351858, 0,
    -2.72161, -0.518451, 1.17353, 0, 1.18346, 0,
    -0.527792, 1.26412, 2.57812, 0, 1.4433, 0,
    1.61107, 3.08528, -0.846102, 0, 0.670558, 0,
    0.827085, 2.41719, -1.42811, 0, 0.954617, 0,
    -2.0954, -0.0847138, 2.49855, 0, 1.66381, 0,
    3.05766, -0.0378456, -1.46936, 0, 0.436099, 0,
    2.60009, 0.186908, -0.223393, 0, 1.71147, 0,
    -3.04189, 1.1841, 2.31376, 0, 1.24432, 0,
    -1.21692, -0.936099, 0.0834011, 0, 1.18667, 0,
    0.234388, -0.201581, -1.33698, 0, 0.567492, 0,
    -0.0095497, 2.86112, 1.56007, 0, 1.13188, 0,
    1.34906, -2.32209, -2.56988, 0, 0.711882, 0,
    -1.63419, -2.00499, -1.14643, 0, 1.63049, 0,
    -0.00162714, -2.21462, 0.54781, 0, 1.56836, 0,
    -0.573236, -2.25051, 0.40777, 0, 0.67819, 0,
    -1.13569, 0.812222, -2.34544, 0, 1.27688, 0,
    -1.86453, -2.96331, 2.52379, 0, 0.939746, 0,
    -2.56245, -2.12247, -2.69509, 0, 0.848009, 0,
    -0.299707, 1.94116, 2.71229, 0, 1.27747, 0,
    -0.180564, 0.0374219, 0.630792, 0, 1.52634, 0,
    2.04013, 1.1874, 1.2705, 0, 1.78072, 0,
    -2.4917, -0.540175, 0.482025, 0, 1.61485, 0,
    1.2977, 1.51875, -3.02163, 0, 1.62905, 0,
    1.05356, 1.14385, -1.88775, 0, 1.67495, 0,
    3.07476, -1.78737, -0.339145, 0, 0.773599, 0,
    -0.847237, -1.81258, 3.13604, 0, 0.530407, 0,
    1.42839, -1.13614, -0.516957, 0, 1.32374, 0,
    -2.54753, -2.62802, 1.65879, 0, 1.24436, 0,
    2.81201, -0.69207, -1.45007, 0, 1.33825, 0,
    -1.92286, -3.07049, -1.93633, 0, 1.77485, 0,
    -2.03022, 2.06311, -2.15054, 0, 1.78191, 0,
    1.23155, 1.85209, 1.23303, 0, 1.42941, 0,
    -1.13865, 1.2556, -2.40371, 0, 1.44386, 0,
    -2.24281, -2.12568, -0.0922046, 0, 1.59034, 0,
    0.179374, -1.18916, 0.553669, 0, 1.07713, 0,
    -0.151448, -0.704777, -1.38674, 0, 0.417395, 0,
    1.43639, 1.38713, 2.79446, 0, 0.991046, 0,
    -0.616802, 0.663754, 3.05679, 0, 0.525591, 0,
    -3.04766, -1.41868, 0.898363, 0, 1.1192, 0,
    1.21909, -0.278021, 1.99241, 0, 0.333126, 0,
    1.58154, 3.08839, 1.22949, 0, 0.719268, 0,
    -2.11692, 2.53402, 1.77809, 0, 1.42002, 0,
    0.276603, -0.692838, -1.78552, 0, 0.878654, 0,
    1.98506, -0.804815, 1.16931, 0, 0.765253, 0,
    -2.79427, -2.69778, -2.13456, 0, 1.38565, 0,
    -0.336216, -2.19716, -1.34496, 0, 1.55052, 0,
    3.05918, 0.266364, -3.13118, 0, 1.58429, 0,
    1.12678, 0.227871, -2.91799, 0, 0.48376, 0,
    -2.35033, 0.332143, 2.85275, 0, 0.835387, 0,
    1.96694, 2.50017, -1.60481, 0, 1.44093, 0,
    -1.82487, -2.42735, 0.295152, 0, 0.309119, 0,
    -0.78345, 2.11245, -2.36476, 0, 1.76002, 0,
    -1.56982, -0.881796, 1.69089, 0, 1.04993, 0,
    -1.69287, -1.74882, 0.39334, 0, 1.27893, 0,
    -2.09904, 1.53375, -2.13781, 0, 0.371458, 0,
    2.6014, -3.02884, 0.587765, 0, 1.38281, 0,
    -2.73095, -0.359459, 2.99821, 0, 1.00161, 0,
    -0.800982, 2.76288, 2.92075, 0, 0.682436, 0,
    -2.4807, 2.14619, -0.825848, 0, 0.935836, 0,
    2.11491, 1.25532, -0.697055, 0, 1.70286, 0,
    -1.63086, -2.61955, -0.492736, 0, 1.00857, 0,
    1.06073, 2.28483, -1.61006, 0, 1.36735, 0,
    1.85576, -0.054393, -3.12131, 0, 0.699179, 0,
    1.85892, 2.80258, -2.06619, 0, 1.209, 0,
    2.10071, 1.42278, -1.0717, 0, 1.50313, 0,
    -2.75372, 0.121286, 2.69643, 0, 0.630773, 0,
    -1.49679, 1.32934, -0.764248, 0, 0.600967, 0,
    -0.348596, -2.92644, 0.149362, 0, 0.347458, 0,
    0.229514, -0.435937, -0.605818, 0, 1.78099, 0,
    3.1265, 0.876651, -0.200592, 0, 1.69787, 0,
    2.79759, 2.019, -2.1651, 0, 0.361612, 0,
    0.386314, 2.24232, 0.106237, 0, 1.31179, 0,
    -0.968416, -2.72508, -2.25748, 0, 1.68168, 0,
    1.20088, 1.64836, 1.34269, 0, 0.415986, 0,
    2.08134, 2.63414, 0.638608, 0, 1.38575, 0,
    -1.42471, 0.197464, 1.24758, 0, 1.30057, 0,
    0.449767, 0.5585, -0.36874, 0, 0.526825, 0,
    -1.12694, -2.92131, -1.60132, 0, 0.460771, 0,
    1.70922, 0.0826144, -0.0836848, 0, 1.27519, 0,
    0.214876, -1.40337, 0.663116, 0, 0.720156, 0,
    0.612461, 1.77816, 2.64052, 0, 1.32224, 0,
    -1.56186, 0.951994, -3.11134, 0, 1.7039, 0,
    -2.36908, -0.5248, 1.27217, 0, 0.954737, 0,
    -1.27508, 1.61868, -1.01467, 0, 0.796138, 0,
    2.98042, 2.29091, -0.0512661, 0, 0.851092, 0,
    -0.200425, -0.75016, 2.41089, 0, 0.9362, 0,
    2.49855, 2.63575, 2.52093, 0, 1.45507, 0,
    2.27214, -1.39991, 2.3216, 0, 1.18029, 0,
    0.501791, 1.56542, 2.39254, 0, 0.807127, 0,
    1.32018, 2.3817, -0.883545, 0, 0.43478, 0,
    1.96693, 2.37625, 1.69244, 0, 1.25281, 0,
    -0.737523, 1.17859, -2.37668, 0, 0.415233, 0,
    2.99005, 0.809065, 1.13851, 0, 1.66304, 0,
    -0.00860443, -0.10133, -0.309542, 0, 1.04949, 0,
    -2.63691, 2.96866, -0.576938, 0, 0.657616, 0,
    -1.01084, 0.475506, -0.375307, 0, 1.17851, 0,
    -2.20673, 1.06938, 3.11269, 0, 1.3471, 0,
    -3.08122, -0.127055, 0.873727, 0, 1.26974, 0}; //vector 6d.

double StartVel[] = {-11.2416, -18.1182, 7.15459, 1.07578, 0, 2.60816,
    -17.8615, 1.18801, 6.84598, -2.95381, 0, -0.699506,
    17.2175, 13.8467, 1.07715, -2.44821, 0, 0.923514,
    -9.50188, -18.1014, 9.44328, -1.03059, 0, 0.795831,
    19.302, 8.90642, 10.1342, 0.909111, 0, -2.56388,
    10.6598, -0.890729, -10.489, -1.35056, 0, -0.84441,
    -17.5774, 16.1861, 0.180916, 0.0977518, 0, -1.0858,
    17.9106, -17.05, 0.0282838, -0.695147, 0, -1.33751,
    -17.9966, 10.4606, 10.8082, 1.9669, 0, -2.24781,
    9.44898, 9.01648, 19.9783, 2.33143, 0, -1.60083,
    13.8393, -3.51677, 13.6604, -1.3841, 0, -0.507632,
    -13.8512, 2.86619, 12.0962, -2.80168, 0, 0.206699,
    15.6295, 4.99397, 13.6816, -2.04139, 0, -1.72349,
    -19.88, -3.42827, -18.9249, 1.25892, 0, 2.62738,
    6.08235, -13.9866, 7.25385, -0.685112, 0, -0.673648,
    3.60434, 18.2164, 2.24585, -2.11109, 0, 2.89983,
    -0.459418, -1.43878, 18.4438, -2.24382, 0, -1.80146,
    4.86536, 12.1229, -10.0863, -0.141409, 0, -0.664115,
    -14.3192, 17.8995, -3.58748, -2.21287, 0, 2.31389,
    -9.87771, -14.5956, 11.3261, -0.268156, 0, -0.902855,
    -11.3901, 7.18369, 16.3569, -1.49925, 0, 2.16516,
    10.2337, -1.5102, 18.0547, 0.796432, 0, -0.364018,
    18.1766, 14.0508, -8.42735, 0.224554, 0, 0.086608,
    -2.39845, 9.18991, 14.7705, 1.29385, 0, 1.80432,
    0.999496, -1.46709, -17.3922, 1.28053, 0, -0.066341,
    14.6353, 15.6007, 1.75793, -2.16483, 0, -0.297915,
    0.586378, 15.2602, -2.41098, -0.194809, 0, 1.8399,
    5.21953, 4.65401, -19.9762, -2.99473, 0, 1.64011,
    7.22248, -11.7895, 13.4568, 1.25352, 0, 1.97225,
    -11.4459, -11.4581, -16.7576, -0.667061, 0, 2.71296,
    -8.6386, 11.0746, 11.3546, -0.465241, 0, -1.30706,
    -10.2378, 12.789, -14.5418, -0.611138, 0, 0.606061,
    -9.71325, -10.656, -15.9345, -1.68354, 0, 0.808305,
    6.78083, 5.3372, -17.7424, 0.5893, 0, -1.63795,
    1.04493, 2.15645, 3.51956, -1.022, 0, 1.21794,
    12.5306, 2.27343, 9.55986, -1.10396, 0, -2.18801,
    -2.76609, -9.64629, -5.19095, -0.641894, 0, -0.318804,
    -5.21032, -9.84311, 6.87137, 1.05742, 0, 0.0836182,
    17.6065, -7.1376, -1.58263, 0.102872, 0, 0.968133,
    6.80394, -6.2473, 1.66841, 0.136846, 0, 1.97543,
    16.7139, -9.33547, 18.8035, -1.5196, 0, 2.06385,
    -13.57, 8.278, 8.31305, -0.380169, 0, 0.494527,
    10.1545, -13.1528, -19.8298, -0.0258535, 0, -2.5205,
    -2.54297, -19.7463, 3.39149, 0.110871, 0, -2.5989,
    5.07445, 6.36213, 8.37279, 0.231968, 0, -1.31827,
    -19.6261, -15.8481, 0.838403, 1.65528, 0, -1.67589,
    -8.11912, -17.9759, -0.887827, -0.257029, 0, 0.119931,
    7.8255, 3.11104, 7.24658, -1.01295, 0, -2.65515,
    -18.8258, -4.89798, -0.413932, 0.457038, 0, 1.43375,
    18.9342, -12.469, -5.94725, 0.684753, 0, 0.645669,
    13.5241, 18.9701, -9.63943, -1.49083, 0, -0.453985,
    7.79426, -1.92595, -9.45519, 0.993021, 0, -2.30057,
    -12.961, 3.97437, -2.72367, -2.50496, 0, 1.12449,
    -18.8146, -16.7859, -0.230215, 1.61662, 0, 2.60418,
    9.97006, 6.87614, 7.26663, 1.54062, 0, -2.78179,
    4.66442, -5.14137, -11.008, -1.59785, 0, 0.91807,
    6.47974, -15.0022, 17.9663, -0.0949172, 0, 0.726274,
    -0.13672, -17.8522, -2.3347, 0.115049, 0, 1.63166,
    -6.83776, -2.16281, 9.73096, -1.76795, 0, -1.97172,
    -17.1866, -15.046, 1.05351, -2.0395, 0, 0.106243,
    -6.9109, 8.54186, 3.07103, 2.23131, 0, 1.54947,
    4.2241, -5.51665, 1.58019, -0.269566, 0, -0.589847,
    -8.40934, -15.8012, -11.3446, 1.81211, 0, 0.164635,
    -9.45348, -4.68714, -16.7598, -0.241423, 0, -1.6028,
    6.58755, -2.99236, -12.527, 2.78085, 0, -2.19577,
    -17.4479, -6.10033, -8.30128, 0.0579494, 0, 1.95502,
    5.27053, -18.1864, -18.3432, -2.08011, 0, 1.60239,
    -11.8175, -16.4403, 7.98764, 1.23877, 0, -0.0322145,
    -10.6957, -2.55615, -1.22006, 2.16712, 0, 2.86343,
    10.127, 3.66353, 12.9156, -1.02253, 0, -1.60753,
    18.6764, 14.0212, 14.7006, -1.13037, 0, -2.10966,
    7.6309, 12.6163, 1.66139, 0.436839, 0, -2.0447,
    -6.25546, -15.5992, -16.1385, 0.11264, 0, -2.86557,
    9.487, 7.93841, -19.088, -1.8481, 0, 1.02595,
    -11.1337, -4.79542, 3.44063, -1.99612, 0, -2.84196,
    3.59237, 17.0201, 17.1879, -0.417324, 0, 0.0358177,
    0.773465, -0.36732, -13.5432, 2.85669, 0, 0.446162,
    -16.1027, 1.94945, 4.42951, 1.00779, 0, -0.029248,
    0.0909891, 9.25455, -18.7337, -2.6903, 0, 0.132381,
    -4.68126, 2.07318, 3.88674, 0.666663, 0, 2.61151,
    7.87983, -3.69849, -0.526019, -0.121303, 0, 1.2627,
    8.13478, 1.21958, 17.5302, -1.58758, 0, -0.519035,
    12.8471, 1.09652, -10.7892, -2.1379, 0, 2.39713,
    -8.98588, 14.3506, -8.84286, 2.71459, 0, 0.0409197,
    -15.6904, 10.647, -16.0054, -0.352409, 0, -0.932541,
    -17.7758, 2.92732, -0.588558, -1.78511, 0, -2.31315,
    12.1965, -13.5118, -13.6168, -2.65366, 0, -2.10449,
    -3.51008, 6.05178, -7.6669, -2.63915, 0, 1.8865,
    -13.1682, 2.89768, -18.7438, 1.81961, 0, 0.133288,
    -4.43312, 12.4758, 1.02681, 2.63923, 0, -0.479772,
    -12.4181, 9.72504, 8.75678, 2.27497, 0, -2.56279,
    -13.5588, -3.32815, -16.1429, 1.04816, 0, 0.37567,
    15.861, 15.1365, -0.541172, -2.32158, 0, -0.735093,
    -14.2679, -0.894369, 8.34352, -1.57188, 0, -0.6542,
    -5.50399, 14.3685, 11.7711, -0.42676, 0, -2.55893,
    9.63035, 17.3137, -7.99513, -2.12283, 0, -2.33989,
    18.5264, 13.339, -12.0768, -2.22311, 0, -1.82216,
    -2.68491, -5.27189, -4.70732, 0.600796, 0, -0.423163,
    -3.8904, 13.9711, 12.7811, 1.76029, 0, -0.754504,
    5.79592, 11.9614, -4.90544, -0.853969, 0, -0.662997}; //vector 6d

class MyWindow : public dart::gui::SimWindow
{
public:
    
    MyWindow(const WorldPtr& world)
    {
        setWorld(world);
        ts = 0;
        sampleCount = 0;
        r1net.load("rect-pdd-r1");
        r2net.load("rect-pdd-r2");
        r3net.load("rect-pdd-r3");
        hand_bd = mWorld->getSkeleton("hand skeleton")->getBodyNode(0);
        ground_bd = mWorld->getSkeleton("ground skeleton")->getBodyNode(0);
        gravity = mWorld->getGravity();
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
        
        if (ts % 800 == 0 && sampleCount < NUMSAM)
        {
            // Record end linear position and orientation in quaternion
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
            // Initialize position
            int curr_index = sampleCount * 6;
            double pos_0=StartPos[curr_index];double pos_1=StartPos[curr_index+1];double pos_2=StartPos[curr_index+2];
            double pos_3=StartPos[curr_index+3];double pos_4=StartPos[curr_index+4];
            double pos_5=StartPos[curr_index+5];
            Eigen::Vector6d InputPos; InputPos<<pos_0,pos_1,pos_2,pos_3,pos_4,pos_5;
            mWorld->getSkeleton("hand skeleton")->getJoint(0)->setPositions(InputPos);
            
            // Initialize velocity
            double vel_0=StartVel[curr_index];double vel_1=StartVel[curr_index+1];double vel_2=StartVel[curr_index+2];
            double vel_3=StartVel[curr_index+3];double vel_4=StartVel[curr_index+4];
            double vel_5=StartVel[curr_index+5];
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
        else if (ts % 800 == 0 && sampleCount == NUMSAM)
        {
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
        bool vel_in_range = checkVelocityRange(vel_uncons);
        Eigen::Matrix3d rm = hand_bd->getWorldTransform().linear(); // rotation matrix
        Eigen::Map<Eigen::VectorXd> rm_flat(rm.data(), rm.size());
        Eigen::VectorXd in_vec = Eigen::VectorXd::Zero(21);
        in_vec.head<9>() = rm_flat;
        in_vec.segment<6>(9) = hand_bd->getSpatialVelocity(Frame::World(),hand_bd);
        auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
        auto collisionGroup = mWorld->getConstraintSolver()->getCollisionGroup();
        dart::collision::CollisionOption option;
        dart::collision::CollisionResult result;
        bool collision = collisionGroup->collide(option, &result);
        if (collision)
        {
            myContactSolve(in_vec, result, vel_in_range);
            if(firstContact == true)
            {
                firstContact = false;
                Eigen::Vector6d firstImp = hand_bd->getConstraintImpulse();
                if (sampleCount < NUMSAM){FirstContactImpulse.push_back(firstImp);}
            }
            //mWorld->getConstraintSolver()->solve();
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
            skel->clearConstraintImpulses();
            skel->resetCommands();
        }
        ts++;
    }

    // Yifeng: they share the same vel range now...
    bool checkVelocityRange(Eigen::Vector6d vel_in)
    {
        //Yifeng: please change below to Vector6d indexing
        // also change to our new range
        // vel[0] = dart::math::random(-20,20);
        // vel[1] = dart::math::random(-20,20);
        // vel[2] = dart::math::random(-20,20);
        // vel[3] = dart::math::random(-3,3);
        // vel[5] = dart::math::random(-3,3);
        // vel[4] = dart::math::random(-6,3);

        if (vel_in[0]<=20 && vel_in[0]>=-20 && vel_in[1]<=20 && vel_in[1]>=-20 && vel_in[2]<=20 && vel_in[2]>=-20)
        {
            if (vel_in[3]<=3 && vel_in[3]>=-3 && vel_in[5]<=3 && vel_in[5]>=-3 && vel_in[4]<=3 && vel_in[4]>=-6)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }

    void scaleInVec(Eigen::VectorXd& in_vec)
    {
        // Yifeng: scale input vector here
        // xall[:, 9:12] /= 30.0
        // xall[:, 12:18] /= 7.0
        // xall[:, 18] *= 6.0
        // xall[:, 20] *= 6.0
        // xall[:, 19] = xall[:, 19] * 10.0 - 1.0
        // vel_in.size() == 21
        
        for (int i = 0; i < in_vec.size(); i++)
        {
            if(i>=9 && i <=11)
            {
                in_vec[i] /= 30.0;
            }
            else if (i >= 12 && i <= 17)
            {
                in_vec[i] /= 7.0;
            }
            else if (i == 18 || i == 20)
            {
                in_vec[i] *= 6.0;
            }
            else if(i == 19)
            {
                in_vec[i] = in_vec[i] * 10.0 - 1.0;
            }
        }
    }
    
    // Yifeng: pass by reference: in_vec
    void myContactSolve(Eigen::VectorXd& in_vec, const dart::collision::CollisionResult& result, bool vel_check)
    {
        if (!vel_check)
        {
            std::cout << "warning: PRE-CONTACT VEL OOR " 
                << hand_bd->getSpatialVelocity(Frame::World(),Frame::World()).transpose() << std::endl;
            mWorld->getConstraintSolver()->solve();
            return;
        }
        
        if (result.getNumContacts() == 1) // PDD R1
        {
            auto PP1 = result.getContact(0).point;
            auto lP1 =  hand_bd->getWorldTransform().inverse() * PP1;
            auto cVel =  hand_bd->getLinearVelocity(lP1,Frame::World(),Frame::World());
            auto con_center = hand_bd->getCOM() - PP1;
            in_vec.segment<3>(15) = cVel;
            in_vec.segment<3>(18) = con_center;
            scaleInVec(in_vec);
            vec_t input_r;
            input_r.assign(in_vec.data(), in_vec.data()+21);
//            for (int j = 0; j < input_r.size(); j++) {
//                cout<<input_r[j]<<" ";
//            }
//            cout<<" "<<endl;
//            cout<<"Point Contact input done!!!!"<<endl;
            vec_t r1output = r1net.predict(input_r);
            // scaled down 100 times when training
            double fx = *(r1output.begin()) * 100.0;
            double fy = *(r1output.begin()+1) * 100.0;
            double fz = *(r1output.begin()+2) * 100.0;
            Eigen::Vector3d con_force;
            con_force << fx, fy, fz;
//            cout<<"Point Contact Force is "<<con_force.transpose()<<endl;
            hand_bd -> clearExternalForces();
            hand_bd -> addExtForce(con_force, lP1, false, true);
            hand_bd -> addConstraintImpulse(hand_bd->getAspectState().mFext * mWorld->getTimeStep());
            hand_bd -> clearExternalForces();
        }
        else if (result.getNumContacts() == 2) // PDD R2
        {
            auto PP1 = result.getContact(0).point;
            auto PP2 = result.getContact(1).point;
            auto cVel = hand_bd->getLinearVelocity(hand_bd->getWorldTransform().inverse() * ((PP1+PP2)/2.0),Frame::World(),Frame::World());
            auto con_center = hand_bd->getCOM() - ((PP1+PP2)/2.0);
            in_vec.segment<3>(15) = cVel;
            in_vec.segment<3>(18) = con_center;
            scaleInVec(in_vec);
            vec_t input_r;
            input_r.assign(in_vec.data(), in_vec.data()+21);
//            for (int j = 0; j < input_r.size(); j++) {
//                cout<<input_r[j]<<" ";
//            }
//            cout<<" "<<endl;
//            cout<<"Line Contact input done!!!!"<<endl;
            vec_t r2output = r2net.predict(input_r);
            // scaled down 100 times when training
            double fx = *(r2output.begin()+3) * 100.0;
            double fy = *(r2output.begin()+4) * 100.0;
            double fz = *(r2output.begin()+5) * 100.0;
            double tx = *(r2output.begin()) * 6.0;
            double ty = *(r2output.begin()+1) * 6.0;
            double tz = *(r2output.begin()+2) * 6.0;
            Eigen::Vector3d con_force;
            con_force << fx, fy, fz;
//            cout<<"Line Contact Force is: "<<con_force.transpose()<<endl;
            Eigen::Vector3d con_torque;
            con_torque << tx, ty, tz;
//            cout<<"Line Contact Torque is: "<<con_torque.transpose()<<endl;
            hand_bd -> clearExternalForces();
            hand_bd -> addExtForce(con_force, Eigen::Vector3d::Zero(), false, true);
            hand_bd -> addExtTorque(con_torque, true);  // Yifeng: PDD outputs local torque
            hand_bd -> addConstraintImpulse(hand_bd->getAspectState().mFext * mWorld->getTimeStep());
            hand_bd -> clearExternalForces();
        }
        else if (result.getNumContacts() == 4) // PDD R3
        {
            //mWorld->getConstraintSolver()->solve();
            auto PP1 = result.getContact(0).point;
            auto PP3 = result.getContact(2).point;
            auto cVel = hand_bd->getLinearVelocity(hand_bd->getWorldTransform().inverse() * ((PP1+PP3)/2.0),Frame::World(),Frame::World());
            auto con_center = hand_bd->getCOM() - ((PP1+PP3)/2.0);
            in_vec.segment<3>(15) = cVel;
            in_vec.segment<3>(18) = con_center;
            scaleInVec(in_vec);

            vec_t input_r;
            input_r.assign(in_vec.data(), in_vec.data()+21);
//            for (int j = 0; j < input_r.size(); j++) {
//                cout<<input_r[j]<<" ";
//            }
//            cout<<" "<<endl;
//            cout<<"Surface input done!!!!"<<endl;
            
            vec_t r3output = r3net.predict(input_r);
            // scaled down 100 times when training
            double fx = *(r3output.begin()+3) * 100.0;
            double fy = *(r3output.begin()+4) * 100.0;
            double fz = *(r3output.begin()+5) * 100.0;
            double tx = *(r3output.begin()) * 6.0;
            double ty = *(r3output.begin()+1) * 6.0;
            double tz = *(r3output.begin()+2) * 6.0;
            Eigen::Vector3d con_force;
            con_force << fx, fy, fz;
            Eigen::Vector3d con_torque;
            con_torque << tx, ty, tz;
//            cout<<"Face Contact Force Is: "<<con_force.transpose()<<endl;
//            cout<<"Face Contact Torque is: "<<con_torque.transpose()<<endl;
            hand_bd -> clearExternalForces();
            hand_bd -> addExtForce(con_force, Eigen::Vector3d::Zero(), false, true);
            hand_bd -> addExtTorque(con_torque, true);  // Yifeng: PDD outputs local torque
            hand_bd -> addConstraintImpulse(hand_bd->getAspectState().mFext * mWorld->getTimeStep());
            hand_bd -> clearExternalForces();
        }
    }
    
    int ts;
    Eigen::Vector6d vel_uncons;
    Eigen::Vector6d pos;
    network<sequential> r1net;
    network<sequential> r2net;
    network<sequential> r3net;
    Eigen::Vector3d gravity;
    BodyNodePtr hand_bd;
    BodyNodePtr ground_bd;

    int sampleCount;
    bool firstContact = true;
    vector<Eigen::Vector3d> EndLinearPos;
    vector<Eigen::Quaterniond> EndAngularPos;
    vector<Eigen::Vector6d> FirstContactImpulse;
    
protected:
};


int main(int argc, char* argv[])
{
    WorldPtr world = SkelParser::readWorld(DART_DATA_PATH"/NN-contact-force/skel/singleBody_test.skel");
    assert(world != nullptr);
    world->setGravity(Eigen::Vector3d(0.0, -10.0, 0));
    MyWindow window(world);
    std::cout << "space bar: simulation on/off" << std::endl;
    glutInit(&argc, argv);
    window.initWindow(640, 480, "Simple Test");
    glutMainLoop();
}
