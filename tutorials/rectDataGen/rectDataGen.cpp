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
#define NUMSAM 100

double GivenStartPos[] =
{-2.31512, 1.60602, -0.259809, 0, 0.45983, 0,
    -0.731978, 0.121997, 2.07952, 0, 0.310372, 0,
    -2.72161, -0.518451, 1.17353, 0, 0.476693, 0,
    -0.527792, 1.26412, 2.57812, 0, 0.528659, 0,
    1.61107, 3.08528, -0.846102, 0, 0.374112, 0,
    0.827085, 2.41719, -1.42811, 0, 0.430923, 0,
    -2.0954, -0.0847138, 2.49855, 0, 0.572762, 0,
    3.05766, -0.0378456, -1.46936, 0, 0.32722, 0,
    2.60009, 0.186908, -0.223393, 0, 0.582294, 0,
    -3.04189, 1.1841, 2.31376, 0, 0.488863, 0,
    -1.21692, -0.936099, 0.0834011, 0, 0.477334, 0,
    0.234388, -0.201581, -1.33698, 0, 0.353498, 0,
    -0.0095497, 2.86112, 1.56007, 0, 0.466375, 0,
    1.34906, -2.32209, -2.56988, 0, 0.382376, 0,
    -1.63419, -2.00499, -1.14643, 0, 0.566097, 0,
    -0.00162714, -2.21462, 0.54781, 0, 0.553673, 0,
    -0.573236, -2.25051, 0.40777, 0, 0.375638, 0,
    -1.13569, 0.812222, -2.34544, 0, 0.495376, 0,
    -1.86453, -2.96331, 2.52379, 0, 0.427949, 0,
    -2.56245, -2.12247, -2.69509, 0, 0.409602, 0,
    -0.299707, 1.94116, 2.71229, 0, 0.495494, 0,
    -0.180564, 0.0374219, 0.630792, 0, 0.545268, 0,
    2.04013, 1.1874, 1.2705, 0, 0.596144, 0,
    -2.4917, -0.540175, 0.482025, 0, 0.56297, 0,
    1.2977, 1.51875, -3.02163, 0, 0.565809, 0,
    1.05356, 1.14385, -1.88775, 0, 0.57499, 0,
    3.07476, -1.78737, -0.339145, 0, 0.39472, 0,
    -0.847237, -1.81258, 3.13604, 0, 0.346081, 0,
    1.42839, -1.13614, -0.516957, 0, 0.504748, 0,
    -2.54753, -2.62802, 1.65879, 0, 0.488872, 0,
    2.81201, -0.69207, -1.45007, 0, 0.507651, 0,
    -1.92286, -3.07049, -1.93633, 0, 0.594971, 0,
    -2.03022, 2.06311, -2.15054, 0, 0.596381, 0,
    1.23155, 1.85209, 1.23303, 0, 0.525882, 0,
    -1.13865, 1.2556, -2.40371, 0, 0.528771, 0,
    -2.24281, -2.12568, -0.0922046, 0, 0.558068, 0,
    0.179374, -1.18916, 0.553669, 0, 0.455425, 0,
    -0.151448, -0.704777, -1.38674, 0, 0.323479, 0,
    1.43639, 1.38713, 2.79446, 0, 0.438209, 0,
    -0.616802, 0.663754, 3.05679, 0, 0.345118, 0,
    -3.04766, -1.41868, 0.898363, 0, 0.463841, 0,
    1.21909, -0.278021, 1.99241, 0, 0.306625, 0,
    1.58154, 3.08839, 1.22949, 0, 0.383854, 0,
    -2.11692, 2.53402, 1.77809, 0, 0.524004, 0,
    0.276603, -0.692838, -1.78552, 0, 0.415731, 0,
    1.98506, -0.804815, 1.16931, 0, 0.393051, 0,
    -2.79427, -2.69778, -2.13456, 0, 0.517129, 0,
    -0.336216, -2.19716, -1.34496, 0, 0.550105, 0,
    3.05918, 0.266364, -3.13118, 0, 0.556858, 0,
    1.12678, 0.227871, -2.91799, 0, 0.336752, 0,
    -2.35033, 0.332143, 2.85275, 0, 0.407077, 0,
    1.96694, 2.50017, -1.60481, 0, 0.528185, 0,
    -1.82487, -2.42735, 0.295152, 0, 0.301824, 0,
    -0.78345, 2.11245, -2.36476, 0, 0.592004, 0,
    -1.56982, -0.881796, 1.69089, 0, 0.449987, 0,
    -1.69287, -1.74882, 0.39334, 0, 0.495787, 0,
    -2.09904, 1.53375, -2.13781, 0, 0.314292, 0,
    2.6014, -3.02884, 0.587765, 0, 0.516561, 0,
    -2.73095, -0.359459, 2.99821, 0, 0.440322, 0,
    -0.800982, 2.76288, 2.92075, 0, 0.376487, 0,
    -2.4807, 2.14619, -0.825848, 0, 0.427167, 0,
    2.11491, 1.25532, -0.697055, 0, 0.580572, 0,
    -1.63086, -2.61955, -0.492736, 0, 0.441714, 0,
    1.06073, 2.28483, -1.61006, 0, 0.513469, 0,
    1.85576, -0.054393, -3.12131, 0, 0.379836, 0,
    1.85892, 2.80258, -2.06619, 0, 0.4818, 0,
    2.10071, 1.42278, -1.0717, 0, 0.540625, 0,
    -2.75372, 0.121286, 2.69643, 0, 0.366155, 0,
    -1.49679, 1.32934, -0.764248, 0, 0.360193, 0,
    -0.348596, -2.92644, 0.149362, 0, 0.309492, 0,
    0.229514, -0.435937, -0.605818, 0, 0.596197, 0,
    3.1265, 0.876651, -0.200592, 0, 0.579574, 0,
    2.79759, 2.019, -2.1651, 0, 0.312322, 0,
    0.386314, 2.24232, 0.106237, 0, 0.502357, 0,
    -0.968416, -2.72508, -2.25748, 0, 0.576335, 0,
    1.20088, 1.64836, 1.34269, 0, 0.323197, 0,
    2.08134, 2.63414, 0.638608, 0, 0.517151, 0,
    -1.42471, 0.197464, 1.24758, 0, 0.500115, 0,
    0.449767, 0.5585, -0.36874, 0, 0.345365, 0,
    -1.12694, -2.92131, -1.60132, 0, 0.332154, 0,
    1.70922, 0.0826144, -0.0836848, 0, 0.495038, 0,
    0.214876, -1.40337, 0.663116, 0, 0.384031, 0,
    0.612461, 1.77816, 2.64052, 0, 0.504447, 0,
    -1.56186, 0.951994, -3.11134, 0, 0.580781, 0,
    -2.36908, -0.5248, 1.27217, 0, 0.430947, 0,
    -1.27508, 1.61868, -1.01467, 0, 0.399228, 0,
    2.98042, 2.29091, -0.0512661, 0, 0.410218, 0,
    -0.200425, -0.75016, 2.41089, 0, 0.42724, 0,
    2.49855, 2.63575, 2.52093, 0, 0.531014, 0,
    2.27214, -1.39991, 2.3216, 0, 0.476059, 0,
    0.501791, 1.56542, 2.39254, 0, 0.401425, 0,
    1.32018, 2.3817, -0.883545, 0, 0.326956, 0,
    1.96693, 2.37625, 1.69244, 0, 0.490562, 0,
    -0.737523, 1.17859, -2.37668, 0, 0.323047, 0,
    2.99005, 0.809065, 1.13851, 0, 0.572607, 0,
    -0.00860443, -0.10133, -0.309542, 0, 0.449897, 0,
    -2.63691, 2.96866, -0.576938, 0, 0.371523, 0,
    -1.01084, 0.475506, -0.375307, 0, 0.475702, 0,
    -2.20673, 1.06938, 3.11269, 0, 0.50942, 0,
    -3.08122, -0.127055, 0.873727, 0, 0.493949, 0};

double GivenStartVel[] =
{-8.43122, -13.5887, 5.36594, 0.896482, 0, 2.17346,
    -13.3962, 0.891006, 5.13448, -2.46151, 0, -0.582922,
    12.9131, 10.385, 0.807863, -2.04018, 0, 0.769595,
    -7.12641, -13.5761, 7.08246, -0.858829, 0, 0.663193,
    14.4765, 6.67981, 7.60068, 0.757593, 0, -2.13657,
    7.99484, -0.668047, -7.86677, -1.12547, 0, -0.703675,
    -13.1831, 12.1396, 0.135687, 0.0814598, 0, -0.904835,
    13.4329, -12.7875, 0.0212128, -0.579289, 0, -1.11459,
    -13.4975, 7.84543, 8.10614, 1.63909, 0, -1.87317,
    7.08674, 6.76236, 14.9837, 1.94286, 0, -1.33403,
    10.3794, -2.63758, 10.2453, -1.15341, 0, -0.423027,
    -10.3884, 2.14964, 9.07217, -2.33473, 0, 0.172249,
    11.7221, 3.74548, 10.2612, -1.70116, 0, -1.43624,
    -14.91, -2.5712, -14.1937, 1.0491, 0, 2.18949,
    4.56176, -10.4899, 5.44039, -0.570926, 0, -0.561373,
    2.70326, 13.6623, 1.68438, -1.75924, 0, 2.41653,
    -0.344564, -1.07908, 13.8329, -1.86985, 0, -1.50121,
    3.64902, 9.09219, -7.56475, -0.117841, 0, -0.553429,
    -10.7394, 13.4246, -2.69061, -1.84406, 0, 1.92824,
    -7.40828, -10.9467, 8.4946, -0.223464, 0, -0.752379,
    -8.54255, 5.38777, 12.2677, -1.24937, 0, 1.8043,
    7.67531, -1.13265, 13.541, 0.663693, 0, -0.303348,
    13.6324, 10.5381, -6.32051, 0.187129, 0, 0.0721733,
    -1.79884, 6.89243, 11.0779, 1.07821, 0, 1.5036,
    0.749622, -1.10032, -13.0442, 1.06711, 0, -0.0552842,
    10.9765, 11.7006, 1.31845, -1.80403, 0, -0.248263,
    0.439784, 11.4451, -1.80823, -0.162341, 0, 1.53325,
    3.91465, 3.49051, -14.9821, -2.49561, 0, 1.36676,
    5.41686, -8.84214, 10.0926, 1.0446, 0, 1.64354,
    -8.58443, -8.5936, -12.5682, -0.555884, 0, 2.2608,
    -6.47895, 8.30597, 8.51596, -0.387701, 0, -1.08922,
    -7.67837, 9.59178, -10.9064, -0.509282, 0, 0.505051,
    -7.28494, -7.99202, -11.9509, -1.40295, 0, 0.673587,
    5.08562, 4.0029, -13.3068, 0.491083, 0, -1.36496,
    0.783698, 1.61734, 2.63967, -0.851668, 0, 1.01495,
    9.39797, 1.70507, 7.1699, -0.919963, 0, -1.82334,
    -2.07457, -7.23472, -3.89321, -0.534912, 0, -0.26567,
    -3.90774, -7.38233, 5.15352, 0.881185, 0, 0.0696819,
    13.2049, -5.3532, -1.18697, 0.0857266, 0, 0.806777,
    5.10295, -4.68547, 1.25131, 0.114039, 0, 1.64619,
    12.5354, -7.0016, 14.1026, -1.26633, 0, 1.71988,
    -10.1775, 6.2085, 6.23479, -0.316808, 0, 0.412106,
    7.61588, -9.86463, -14.8724, -0.0215446, 0, -2.10042,
    -1.90723, -14.8097, 2.54362, 0.0923921, 0, -2.16575,
    3.80584, 4.7716, 6.2796, 0.193306, 0, -1.09856,
    -14.7196, -11.8861, 0.628802, 1.3794, 0, -1.39657,
    -6.08934, -13.4819, -0.66587, -0.214191, 0, 0.0999424,
    5.86912, 2.33328, 5.43493, -0.844125, 0, -2.21262,
    -14.1193, -3.67349, -0.310449, 0.380865, 0, 1.19479,
    14.2007, -9.35173, -4.46044, 0.570628, 0, 0.538057,
    10.143, 14.2276, -7.22957, -1.24236, 0, -0.378321,
    5.84569, -1.44446, -7.09139, 0.827517, 0, -1.91714,
    -9.72077, 2.98078, -2.04275, -2.08747, 0, 0.937072,
    -14.1109, -12.5894, -0.172661, 1.34719, 0, 2.17015,
    7.47755, 5.1571, 5.44997, 1.28385, 0, -2.31816,
    3.49831, -3.85603, -8.25596, -1.33154, 0, 0.765058,
    4.85981, -11.2517, 13.4747, -0.0790977, 0, 0.605228,
    -0.10254, -13.3892, -1.75102, 0.0958743, 0, 1.35972,
    -5.12832, -1.6221, 7.29822, -1.47329, 0, -1.6431,
    -12.8899, -11.2845, 0.790135, -1.69958, 0, 0.0885361,
    -5.18317, 6.4064, 2.30328, 1.85942, 0, 1.29123,
    3.16808, -4.13749, 1.18514, -0.224638, 0, -0.491539,
    -6.30701, -11.8509, -8.50842, 1.51009, 0, 0.137195,
    -7.09011, -3.51535, -12.5698, -0.201186, 0, -1.33567,
    4.94066, -2.24427, -9.39526, 2.31738, 0, -1.8298,
    -13.0859, -4.57525, -6.22596, 0.0482911, 0, 1.62918,
    3.9529, -13.6398, -13.7574, -1.73342, 0, 1.33533,
    -8.86311, -12.3302, 5.99073, 1.03231, 0, -0.0268454,
    -8.02177, -1.91711, -0.915045, 1.80594, 0, 2.38619,
    7.59521, 2.74765, 9.68673, -0.852106, 0, -1.33961,
    14.0073, 10.5159, 11.0254, -0.941974, 0, -1.75805,
    5.72318, 9.4622, 1.24604, 0.364033, 0, -1.70392,
    -4.6916, -11.6994, -12.1039, 0.0938664, 0, -2.38797,
    7.11525, 5.95381, -14.316, -1.54008, 0, 0.854955,
    -8.35031, -3.59656, 2.58047, -1.66344, 0, -2.3683,
    2.69428, 12.7651, 12.8909, -0.34777, 0, 0.0298481,
    0.580099, -0.27549, -10.1574, 2.38058, 0, 0.371801,
    -12.077, 1.46209, 3.32213, 0.839827, 0, -0.0243733,
    0.0682419, 6.94091, -14.0503, -2.24192, 0, 0.110317,
    -3.51094, 1.55488, 2.91506, 0.555553, 0, 2.17626,
    5.90987, -2.77387, -0.394515, -0.101086, 0, 1.05225,
    6.10108, 0.914687, 13.1476, -1.32299, 0, -0.432529,
    9.63532, 0.82239, -8.09191, -1.78158, 0, 1.99761,
    -6.73941, 10.763, -6.63214, 2.26215, 0, 0.0340998,
    -11.7678, 7.98524, -12.004, -0.293674, 0, -0.777118,
    -13.3318, 2.19549, -0.441419, -1.48759, 0, -1.92763,
    9.14737, -10.1339, -10.2126, -2.21139, 0, -1.75374,
    -2.63256, 4.53884, -5.75018, -2.19929, 0, 1.57208,
    -9.87611, 2.17326, -14.0579, 1.51634, 0, 0.111073,
    -3.32484, 9.35686, 0.770107, 2.19936, 0, -0.39981,
    -9.31354, 7.29378, 6.56758, 1.89581, 0, -2.13565,
    -10.1691, -2.49611, -12.1071, 0.873464, 0, 0.313058,
    11.8957, 11.3524, -0.405879, -1.93465, 0, -0.612577,
    -10.7009, -0.670777, 6.25764, -1.3099, 0, -0.545167,
    -4.128, 10.7764, 8.82834, -0.355634, 0, -2.13245,
    7.22276, 12.9853, -5.99635, -1.76902, 0, -1.94991,
    13.8948, 10.0042, -9.0576, -1.85259, 0, -1.51846,
    -2.01368, -3.95392, -3.53049, 0.500663, 0, -0.352636,
    -2.9178, 10.4783, 9.58582, 1.46691, 0, -0.628753,
    4.34694, 8.97104, -3.67908, -0.711641, 0, -0.552498};

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
            c1net.load(DART_DATA_PATH"NN-contact-force/neuralnets/3D-rect-C1");
            c2net.load(DART_DATA_PATH"NN-contact-force/neuralnets/3D-rect-C2-input15");
            c3net.load(DART_DATA_PATH"NN-contact-force/neuralnets/3D-rect-C3");
            r1net.load(DART_DATA_PATH"NN-contact-force/neuralnets/3D-rect-R1");
            r2net.load(DART_DATA_PATH"NN-contact-force/neuralnets/3D-rect-R2");
            r3net.load(DART_DATA_PATH"NN-contact-force/neuralnets/3D-rect-R3");
        }
        else if (conModel == 2)
        {
            // PDD
            r1net.load(DART_DATA_PATH"NN-contact-force/neuralnets/3D-rect-PDD-R1");
            r2net.load(DART_DATA_PATH"NN-contact-force/neuralnets/3D-rect-PDD-R2");
            r3net.load(DART_DATA_PATH"NN-contact-force/neuralnets/3D-rect-PDD-R3");
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
    
    Eigen::Vector3d calculateCntFor3PointBox()
    {
        Eigen::Vector3d dist((result.getContact(0).point-result.getContact(1).point).norm(),
                             (result.getContact(1).point-result.getContact(2).point).norm(),
                             (result.getContact(2).point-result.getContact(0).point).norm());
        if (dist.maxCoeff() == dist[0])
            return (result.getContact(0).point + result.getContact(1).point) / 2.0;
        else if (dist.maxCoeff() == dist[1])
            return (result.getContact(1).point + result.getContact(2).point) / 2.0;
        else
            return (result.getContact(2).point + result.getContact(0).point) / 2.0;
    }
    
    void timeStepping() override
    // substituting original DART contact solve with ours, keeping other steps in world->step() unchanged.
    {
        if(ts % 800 == 0 && sampleCount < NUMSAM)
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
                pos[0] = dart::math::random(-1*Pi, Pi); pos[1] = dart::math::random(-1*Pi, Pi);
                pos[2] = dart::math::random(-1*Pi, Pi); pos[3] = 0.0;
                pos[5] = 0.0;                           pos[4] = dart::math::random(0.3, 0.6);
                if (sampleCount < NUMSAM){StartPos.push_back(pos);}
                mWorld->getSkeleton("hand skeleton")->getJoint(0)->setPositions(pos);
                
                // Set velocities
                Eigen::Vector6d vel = Eigen::Vector6d::Zero();
                vel[0] = dart::math::random(-15,15); vel[1] = dart::math::random(-15,15);
                vel[2] = dart::math::random(-15,15); vel[3] = dart::math::random(-2.5,2.5); //x vel
                vel[4] = 0.0;                        vel[5] = dart::math::random(-2.5,2.5); //z vel
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
        else if (ts % 800 == 0 && sampleCount == NUMSAM)
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
//        cout << "t: " << ts << endl;
//        cout<<"Pos is: "<<pos.transpose()<<endl;
        vel_uncons = hand_bd->getSpatialVelocity(Frame::World(),Frame::World());
//        cout<<"Velocity is: "<<vel_uncons.transpose()<<endl;
        bool vel_in_range = checkVelocityRange(vel_uncons);
        
        in_vec = Eigen::VectorXd::Zero(21);
        
        auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
        auto collisionGroup = mWorld->getConstraintSolver()->getCollisionGroup();
        dart::collision::CollisionOption option;
        //        dart::collision::CollisionResult result; // YIFENG: CAUTION I deleted this line
        result.clear();
        bool collision = collisionGroup->collide(option, &result);
        
        //        Eigen::Vector3d oldAng = hand_bd->getWorldTransform().linear() * hand_bd->getAngularMomentum();
        //cout<<result.getNumContacts()<<endl;
        
        if (collision)
        {
            tempImp = Eigen::Vector6d::Zero();
            
            if (conModel == 0)
            {
                mWorld->getConstraintSolver()->solve();
                
            }
            if (conModel == 1)
            {
                myContactSolve(in_vec, result, vel_in_range);
            }
            
            if (conModel == 2)
            {
                pddContactSolve(in_vec, result, vel_in_range);
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
            
            //            Eigen::Vector3d newVel = hand_bd->getCOMLinearVelocity(Frame::World(),Frame::World());
            //            auto fricx = (newVel[0] -  vel_uncons[3])/mWorld->getTimeStep();
            //            auto fy = (newVel[1] -  vel_uncons[4])/mWorld->getTimeStep();
            //            auto fricz = (newVel[2] -  vel_uncons[5])/mWorld->getTimeStep();
            //            Eigen::Vector3d newAng = hand_bd->getWorldTransform().linear() * hand_bd->getAngularMomentum();
            //            auto tauy = (newAng[1] - oldAng[1])/mWorld->getTimeStep();
            //            std::cout << fricx << " " << fy << " "<< fricz << " " << tauy << std::endl;
            
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
    
    double scaleInVec(Eigen::VectorXd& in_vec, bool sc)
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
        
        Eigen::VectorXd Velrelated = in_vec.segment<9>(9).cwiseAbs();
        double scale = 1.0;
        if (sc && std::abs(vel_uncons[4]) < 0.6)
        {
            scale = 0.8 / Velrelated.maxCoeff();
            in_vec.segment<9>(9) = in_vec.segment<9>(9) * scale;
        }
        
        //        for (int i = 0; i < in_vec.size(); i++)
        //        {
        //            std::cout << in_vec[i] << " ,";
        //        }
        //        std::cout << std::endl;
        
        return scale;
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
        
        Eigen::Matrix3d rm = hand_bd->getWorldTransform().linear(); // rotation matrix
        Eigen::Map<Eigen::VectorXd> rm_flat(rm.data(), rm.size());
        in_vec.head<9>() = rm_flat;
        // Yifeng: we are using local vel as feature
        in_vec.segment<6>(9) = hand_bd->getSpatialVelocity(Frame::World(),hand_bd);
        
        if (result.getNumContacts() == 1) // C1 R1
        {
            auto PP1 = result.getContact(0).point;
            auto lP1 =  hand_bd->getWorldTransform().inverse() * PP1;
            auto cVel =  hand_bd->getLinearVelocity(lP1,Frame::World(),Frame::World());
            auto con_center = hand_bd->getCOM() - PP1;
            in_vec.segment<3>(15) = cVel;
            in_vec.segment<3>(18) = con_center;
            double scale = scaleInVec(in_vec, true);
            
            vec_t inputnn;
            inputnn.assign(in_vec.data(), in_vec.data()+21);
            
            label_t c1output = c1net.predict_label(inputnn);
//            cout << "Point Contact:" << c1output << endl;
            if (c1output == 1) // Static case -> apply constraints
            {
                // set constraints
                auto constraint = std::make_shared<dart::constraint::BallJointConstraint>(hand_bd, ground_bd, PP1);
                mWorld->getConstraintSolver()->addConstraint(constraint);
                
                // disable ground from contact detection
                mWorld->getConstraintSolver()->getCollisionGroup()->
                removeShapeFramesOf(ground_bd->getSkeleton().get());
                
                // solve for impulses needed to maintain ball joints
                mWorld->getConstraintSolver()->solve(); // this will call collide() again...
                
                // restore
                mWorld->getConstraintSolver()->removeConstraint(constraint);
                mWorld->getConstraintSolver()->getCollisionGroup()->
                addShapeFramesOf(ground_bd->getSkeleton().get());
            }
            else if (c1output == 0)
            {
                vec_t r1output = r1net.predict(inputnn);
                // scaled down 100 times when training
                double fx = *(r1output.begin()) * 100.0 / scale;
                double fz = *(r1output.begin()+1) * 100.0 / scale;
                
                Eigen::Vector3d fric_force;
                fric_force << fx, 0, fz;
                
//                cout << scale << endl;
//
//                cout<<"Point Contact Force is "<< fric_force.transpose()<<endl;
                
                hand_bd -> clearExternalForces();
                hand_bd -> addExtForce(fric_force, lP1, false, true);
                
                hand_bd -> addConstraintImpulse(hand_bd->getAspectState().mFext * mWorld->getTimeStep());
//                std::cout << hand_bd -> getConstraintImpulse().transpose() << std::endl;
                hand_bd -> clearExternalForces();
                hand_bd -> getSkeleton()->computeImpulseForwardDynamics();
                
                hand_bd -> setFrictionCoeff(0.0);
                ground_bd -> setFrictionCoeff(0.0);
                
                tempImp = hand_bd->getConstraintImpulse();
                
                hand_bd -> setConstraintImpulse(Eigen::Vector6d::Zero());
                mWorld->getConstraintSolver()->solve();
                
//                std::cout << hand_bd -> getConstraintImpulse().transpose() << std::endl;
                
                hand_bd -> setFrictionCoeff(1.0);
                ground_bd -> setFrictionCoeff(1.0);
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
            }
            
        }
        else if (result.getNumContacts() == 2) // C2 R2
        {
            auto PP1 = result.getContact(0).point;
            auto PP2 = result.getContact(1).point;
            auto cVel = hand_bd->getLinearVelocity(hand_bd->getWorldTransform().inverse() * ((PP1+PP2)/2.0),Frame::World(),Frame::World());
            auto con_center = hand_bd->getCOM() - ((PP1+PP2)/2.0);
            in_vec.segment<3>(15) = cVel;
            in_vec.segment<3>(18) = con_center;
            double scale = scaleInVec(in_vec, true);
            
            // TEMP: use 15 input C2
            vec_t inputnn_c;
            inputnn_c.assign(in_vec.data(), in_vec.data()+15);
            
            vec_t inputnn;
            inputnn.assign(in_vec.data(), in_vec.data()+21);
            //            // TEMP: R2 is not scaled properly...
            //            inputnn[12] *= 0.7; inputnn[13] *= 0.7; inputnn[14] *= 0.7;
            //            inputnn[16] += 0.8;
            
            label_t c2output = c2net.predict_label(inputnn_c);
//            cout << "Line Contact:" << c2output <<endl;
            
            if (c2output == 1)
            {
                auto constraint1 = std::make_shared<dart::constraint::BallJointConstraint>(hand_bd, ground_bd, PP1);
                auto constraint2 = std::make_shared<dart::constraint::BallJointConstraint>(hand_bd, ground_bd, PP2);
                
                mWorld->getConstraintSolver()->addConstraint(constraint1);
                mWorld->getConstraintSolver()->addConstraint(constraint2);
                
                // disable ground from contact detection
                mWorld->getConstraintSolver()->getCollisionGroup()->
                removeShapeFramesOf(ground_bd->getSkeleton().get());
                
                mWorld->getConstraintSolver()->solve();
                
                // restore
                mWorld->getConstraintSolver()->removeConstraint(constraint1);
                mWorld->getConstraintSolver()->removeConstraint(constraint2);
                mWorld->getConstraintSolver()->getCollisionGroup()->
                addShapeFramesOf(ground_bd->getSkeleton().get());
            }
            else if (c2output == 0)
            {
                vec_t r2output = r2net.predict(inputnn);
                // scaled down 100 times when training
                double fx = *(r2output.begin()) * 100.0  / scale;
                double fz = *(r2output.begin()+1) * 100.0  / scale;
                double tauy = *(r2output.begin()+2) * 6.0  / scale;
                
                Eigen::Vector3d fric_force;
                fric_force << fx, 0, fz;
                Eigen::Vector3d torque;
                torque << -pos[4]*fz, tauy, pos[4]*fx;
                
//                cout<<"Line Contact Force/toque is "<< fric_force.transpose() << " t " << tauy << endl;
                
                hand_bd -> clearExternalForces();
                hand_bd -> addExtForce(fric_force, Eigen::Vector3d(0.0, 0.0, 0), false, true);
                hand_bd -> addExtTorque(torque, false);
                
                hand_bd -> addConstraintImpulse(hand_bd->getAspectState().mFext * mWorld->getTimeStep());
                hand_bd -> clearExternalForces();
                hand_bd -> getSkeleton()->computeImpulseForwardDynamics();
                
                hand_bd -> setFrictionCoeff(0.0);
                ground_bd -> setFrictionCoeff(0.0);
                
                tempImp = hand_bd->getConstraintImpulse();
                
                hand_bd -> setConstraintImpulse(Eigen::Vector6d::Zero());
                mWorld->getConstraintSolver()->solve(); // add normal impulse
                
                hand_bd -> setFrictionCoeff(1.0);
                ground_bd -> setFrictionCoeff(1.0);
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
            }
            
        }
        else if (result.getNumContacts() >= 3) // C3 R3
        {
            Eigen::Vector3d cnt = Eigen::Vector3d(0,0,0);;
            if (result.getNumContacts() == 3)
                cnt = calculateCntFor3PointBox();
            else // np = 4
            {
                for (auto contact : result.getContacts())
                {
                    cnt = cnt + contact.point;
                }
                cnt = cnt / result.getNumContacts();
            }
            
            auto cVel = hand_bd->getLinearVelocity(hand_bd->getWorldTransform().inverse() * cnt, Frame::World(),Frame::World());
            auto con_center = hand_bd->getCOM() - cnt;
            in_vec.segment<3>(15) = cVel;
            in_vec.segment<3>(18) = con_center;
            double scale = scaleInVec(in_vec, true);
            
            vec_t inputnn;
            inputnn.assign(in_vec.data(), in_vec.data()+21);
            
            label_t c3output = c3net.predict_label(inputnn);
//            cout << "Face Contact:" << c3output <<endl;
            
            if (c3output == 1)
            {
                auto w_cstr = std::make_shared<dart::constraint::WeldJointConstraint>(hand_bd, ground_bd);
                mWorld->getConstraintSolver()->addConstraint(w_cstr);
                
                // disable ground from contact detection
                mWorld->getConstraintSolver()->getCollisionGroup()->
                removeShapeFramesOf(ground_bd->getSkeleton().get());
                
                mWorld->getConstraintSolver()->solve();
                mWorld->getConstraintSolver()->removeConstraint(w_cstr);
                mWorld->getConstraintSolver()->getCollisionGroup()->
                addShapeFramesOf(ground_bd->getSkeleton().get());
            }
            else if (c3output == 0)
            {
                vec_t r3output = r3net.predict(inputnn);
                // scaled down 100 times when training
                double fx = *(r3output.begin()) * 100.0 / scale;
                double fz = *(r3output.begin()+1) * 100.0 / scale;
                double tauy = *(r3output.begin()+2) * 6.0 / scale;
                
                Eigen::Vector3d fric_force;
                fric_force << fx, 0, fz;
                Eigen::Vector3d torque;
                torque << -pos[4]*fz, tauy, pos[4]*fx;
                
//                cout<<"Face Contact Force/toque is "<< fric_force.transpose() << " t " << tauy << endl;
                
                hand_bd -> clearExternalForces();
                hand_bd -> addExtForce(fric_force, Eigen::Vector3d(0.0, 0.0, 0), false, true);
                hand_bd -> addExtTorque(torque, false);
                
                hand_bd -> addConstraintImpulse(hand_bd->getAspectState().mFext * mWorld->getTimeStep());
                hand_bd -> clearExternalForces();
                hand_bd -> getSkeleton()->computeImpulseForwardDynamics();
                
                hand_bd -> setFrictionCoeff(0.0);
                ground_bd -> setFrictionCoeff(0.0);
                
                tempImp = hand_bd->getConstraintImpulse();
                
                hand_bd -> setConstraintImpulse(Eigen::Vector6d::Zero());
                mWorld->getConstraintSolver()->solve(); // add normal impulse
                
                hand_bd -> setFrictionCoeff(1.0);
                ground_bd -> setFrictionCoeff(1.0);
            }
            else // c3out == 2
            {
                // do nothing, ignore collision solving
                mWorld->getConstraintSolver()->getCollisionGroup()->
                removeShapeFramesOf(ground_bd->getSkeleton().get());
                
                mWorld->getConstraintSolver()->solve();
                // restore
                mWorld->getConstraintSolver()->getCollisionGroup()->
                addShapeFramesOf(ground_bd->getSkeleton().get());
            }
        }
        else
        {
            std::cout << "WARNING: # of contact" << result.getNumContacts() << std::endl;
        }
    }
    
    
    // Yifeng: pass by reference: in_vec
    void pddContactSolve(Eigen::VectorXd& in_vec, const dart::collision::CollisionResult& result, bool vel_check)
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
            double scale = scaleInVec(in_vec, false);
            vec_t input_r;
            input_r.assign(in_vec.data(), in_vec.data()+21);
            
            vec_t r1output = r1net.predict(input_r);
            // scaled down 100 times when training
            double fx = *(r1output.begin()) * 100.0 / scale;
            double fy = *(r1output.begin()+1) * 100.0 / scale;
            double fz = *(r1output.begin()+2) * 100.0 / scale;
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
            double scale = scaleInVec(in_vec, false);
            vec_t input_r;
            input_r.assign(in_vec.data(), in_vec.data()+21);
            
            vec_t r2output = r2net.predict(input_r);
            // scaled down 100 times when training
            double fx = *(r2output.begin()+3) * 100.0 / scale;
            double fy = *(r2output.begin()+4) * 100.0 / scale;
            double fz = *(r2output.begin()+5) * 100.0 / scale;
            double tx = *(r2output.begin()) * 6.0 / scale;
            double ty = *(r2output.begin()+1) * 6.0 / scale;
            double tz = *(r2output.begin()+2) * 6.0 / scale;
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
        else if (result.getNumContacts() >= 3) // PDD R3
        {
            Eigen::Vector3d cnt = Eigen::Vector3d(0,0,0);;
            if (result.getNumContacts() == 3)
                cnt = calculateCntFor3PointBox();
            else // np = 4
            {
                for (auto contact : result.getContacts())
                {
                    cnt = cnt + contact.point;
                }
                cnt = cnt / result.getNumContacts();
            }
            
            auto cVel = hand_bd->getLinearVelocity(hand_bd->getWorldTransform().inverse() * cnt,Frame::World(),Frame::World());
            auto con_center = hand_bd->getCOM() - cnt;
            in_vec.segment<3>(15) = cVel;
            in_vec.segment<3>(18) = con_center;
            double scale = scaleInVec(in_vec, false);
            
            vec_t input_r;
            input_r.assign(in_vec.data(), in_vec.data()+21);
            
            vec_t r3output = r3net.predict(input_r);
            // scaled down 100 times when training
            double fx = *(r3output.begin()+3) * 100.0 / scale;
            double fy = *(r3output.begin()+4) * 100.0 / scale;
            double fz = *(r3output.begin()+5) * 100.0 / scale;
            double tx = *(r3output.begin()) * 6.0 / scale;
            double ty = *(r3output.begin()+1) * 6.0 / scale;
            double tz = *(r3output.begin()+2) * 6.0 / scale;
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
    double theta_odo;
    
    Eigen::Vector6d vel_uncons;
    Eigen::Vector6d pos;
    Eigen::VectorXd in_vec;
    
    network<sequential> c1net;
    network<sequential> c2net;
    network<sequential> c3net;
    network<sequential> r1net;
    network<sequential> r2net;
    network<sequential> r3net;
    
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
    dart::collision::CollisionResult result;
protected:
};


int main(int argc, char* argv[])
{
//    std::cout.precision(10);
    WorldPtr world = SkelParser::readWorld(DART_DATA_PATH"/skel/mytest/hopperfoot2d_forTest.skel");
    assert(world != nullptr);
    world->setGravity(Eigen::Vector3d(0.0, -10.0, 0));
    MyWindow window(world);
    std::cout << "space bar: simulation on/off" << std::endl;
    glutInit(&argc, argv);
    window.initWindow(640, 480, "Simple Test");
    glutMainLoop();
}
