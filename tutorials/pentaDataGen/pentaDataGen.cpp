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

#define NUMSAM 50

double StartPos[] = {-2.31512, 1.60602, -0.259809, 0, 0.379915, 0,
    -0.731978, 0.121997, 2.07952, 0, 0.305186, 0,
    -2.72161, -0.518451, 1.17353, 0, 0.388346, 0,
    -0.527792, 1.26412, 2.57812, 0, 0.41433, 0,
    1.61107, 3.08528, -0.846102, 0, 0.337056, 0,
    0.827085, 2.41719, -1.42811, 0, 0.365462, 0,
    -2.0954, -0.0847138, 2.49855, 0, 0.436381, 0,
    3.05766, -0.0378456, -1.46936, 0, 0.31361, 0,
    2.60009, 0.186908, -0.223393, 0, 0.441147, 0,
    -3.04189, 1.1841, 2.31376, 0, 0.394432, 0,
    -1.21692, -0.936099, 0.0834011, 0, 0.388667, 0,
    0.234388, -0.201581, -1.33698, 0, 0.326749, 0,
    -0.0095497, 2.86112, 1.56007, 0, 0.383188, 0,
    1.34906, -2.32209, -2.56988, 0, 0.341188, 0,
    -1.63419, -2.00499, -1.14643, 0, 0.433049, 0,
    -0.00162714, -2.21462, 0.54781, 0, 0.426836, 0,
    -0.573236, -2.25051, 0.40777, 0, 0.337819, 0,
    -1.13569, 0.812222, -2.34544, 0, 0.397688, 0,
    -1.86453, -2.96331, 2.52379, 0, 0.363975, 0,
    -2.56245, -2.12247, -2.69509, 0, 0.354801, 0,
    -0.299707, 1.94116, 2.71229, 0, 0.397747, 0,
    -0.180564, 0.0374219, 0.630792, 0, 0.422634, 0,
    2.04013, 1.1874, 1.2705, 0, 0.448072, 0,
    -2.4917, -0.540175, 0.482025, 0, 0.431485, 0,
    1.2977, 1.51875, -3.02163, 0, 0.432905, 0,
    1.05356, 1.14385, -1.88775, 0, 0.437495, 0,
    3.07476, -1.78737, -0.339145, 0, 0.34736, 0,
    -0.847237, -1.81258, 3.13604, 0, 0.323041, 0,
    1.42839, -1.13614, -0.516957, 0, 0.402374, 0,
    -2.54753, -2.62802, 1.65879, 0, 0.394436, 0,
    2.81201, -0.69207, -1.45007, 0, 0.403825, 0,
    -1.92286, -3.07049, -1.93633, 0, 0.447485, 0,
    -2.03022, 2.06311, -2.15054, 0, 0.448191, 0,
    1.23155, 1.85209, 1.23303, 0, 0.412941, 0,
    -1.13865, 1.2556, -2.40371, 0, 0.414386, 0,
    -2.24281, -2.12568, -0.0922046, 0, 0.429034, 0,
    0.179374, -1.18916, 0.553669, 0, 0.377713, 0,
    -0.151448, -0.704777, -1.38674, 0, 0.311739, 0,
    1.43639, 1.38713, 2.79446, 0, 0.369105, 0,
    -0.616802, 0.663754, 3.05679, 0, 0.322559, 0,
    -3.04766, -1.41868, 0.898363, 0, 0.38192, 0,
    1.21909, -0.278021, 1.99241, 0, 0.303313, 0,
    1.58154, 3.08839, 1.22949, 0, 0.341927, 0,
    -2.11692, 2.53402, 1.77809, 0, 0.412002, 0,
    0.276603, -0.692838, -1.78552, 0, 0.357865, 0,
    1.98506, -0.804815, 1.16931, 0, 0.346525, 0,
    -2.79427, -2.69778, -2.13456, 0, 0.408565, 0,
    -0.336216, -2.19716, -1.34496, 0, 0.425052, 0,
    3.05918, 0.266364, -3.13118, 0, 0.428429, 0,
    1.12678, 0.227871, -2.91799, 0, 0.318376, 0};

double StartVel[] = {-8.43122, -13.5887, 5.36594, 1.07578, 0, 2.60816,
    -13.3962, 0.891006, 5.13448, -2.95381, 0, -0.699506,
    12.9131, 10.385, 0.807863, -2.44821, 0, 0.923514,
    -7.12641, -13.5761, 7.08246, -1.03059, 0, 0.795831,
    14.4765, 6.67981, 7.60068, 0.909111, 0, -2.56388,
    7.99484, -0.668047, -7.86677, -1.35056, 0, -0.84441,
    -13.1831, 12.1396, 0.135687, 0.0977518, 0, -1.0858,
    13.4329, -12.7875, 0.0212128, -0.695147, 0, -1.33751,
    -13.4975, 7.84543, 8.10614, 1.9669, 0, -2.24781,
    7.08674, 6.76236, 14.9837, 2.33143, 0, -1.60083,
    10.3794, -2.63758, 10.2453, -1.3841, 0, -0.507632,
    -10.3884, 2.14964, 9.07217, -2.80168, 0, 0.206699,
    11.7221, 3.74548, 10.2612, -2.04139, 0, -1.72349,
    -14.91, -2.5712, -14.1937, 1.25892, 0, 2.62738,
    4.56176, -10.4899, 5.44039, -0.685112, 0, -0.673648,
    2.70326, 13.6623, 1.68438, -2.11109, 0, 2.89983,
    -0.344564, -1.07908, 13.8329, -2.24382, 0, -1.80146,
    3.64902, 9.09219, -7.56475, -0.141409, 0, -0.664115,
    -10.7394, 13.4246, -2.69061, -2.21287, 0, 2.31389,
    -7.40828, -10.9467, 8.4946, -0.268156, 0, -0.902855,
    -8.54255, 5.38777, 12.2677, -1.49925, 0, 2.16516,
    7.67531, -1.13265, 13.541, 0.796432, 0, -0.364018,
    13.6324, 10.5381, -6.32051, 0.224554, 0, 0.086608,
    -1.79884, 6.89243, 11.0779, 1.29385, 0, 1.80432,
    0.749622, -1.10032, -13.0442, 1.28053, 0, -0.066341,
    10.9765, 11.7006, 1.31845, -2.16483, 0, -0.297915,
    0.439784, 11.4451, -1.80823, -0.194809, 0, 1.8399,
    3.91465, 3.49051, -14.9821, -2.99473, 0, 1.64011,
    5.41686, -8.84214, 10.0926, 1.25352, 0, 1.97225,
    -8.58443, -8.5936, -12.5682, -0.667061, 0, 2.71296,
    -6.47895, 8.30597, 8.51596, -0.465241, 0, -1.30706,
    -7.67837, 9.59178, -10.9064, -0.611138, 0, 0.606061,
    -7.28494, -7.99202, -11.9509, -1.68354, 0, 0.808305,
    5.08562, 4.0029, -13.3068, 0.5893, 0, -1.63795,
    0.783698, 1.61734, 2.63967, -1.022, 0, 1.21794,
    9.39797, 1.70507, 7.1699, -1.10396, 0, -2.18801,
    -2.07457, -7.23472, -3.89321, -0.641894, 0, -0.318804,
    -3.90774, -7.38233, 5.15352, 1.05742, 0, 0.0836182,
    13.2049, -5.3532, -1.18697, 0.102872, 0, 0.968133,
    5.10295, -4.68547, 1.25131, 0.136846, 0, 1.97543,
    12.5354, -7.0016, 14.1026, -1.5196, 0, 2.06385,
    -10.1775, 6.2085, 6.23479, -0.380169, 0, 0.494527,
    7.61588, -9.86463, -14.8724, -0.0258535, 0, -2.5205,
    -1.90723, -14.8097, 2.54362, 0.110871, 0, -2.5989,
    3.80584, 4.7716, 6.2796, 0.231968, 0, -1.31827,
    -14.7196, -11.8861, 0.628802, 1.65528, 0, -1.67589,
    -6.08934, -13.4819, -0.66587, -0.257029, 0, 0.119931,
    5.86912, 2.33328, 5.43493, -1.01295, 0, -2.65515,
    -14.1193, -3.67349, -0.310449, 0.457038, 0, 1.43375,
    14.2007, -9.35173, -4.46044, 0.684753, 0, 0.645669};

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
            c1net.load(DART_DATA_PATH"NN-contact-force/neuralnets/3D-5gon-C1");
            c2net.load(DART_DATA_PATH"NN-contact-force/neuralnets/3D-5gon-C2");
            c3net.load(DART_DATA_PATH"NN-contact-force/neuralnets/3D-5gon-C3");
            r1net.load(DART_DATA_PATH"NN-contact-force/neuralnets/3D-5gon-R1");
            r2net.load(DART_DATA_PATH"NN-contact-force/neuralnets/3D-5gon-R2");
            r3net.load(DART_DATA_PATH"NN-contact-force/neuralnets/3D-5gon-R3");
        }
        else if (conModel == 2)
        {
            // PDD
            r1net.load(DART_DATA_PATH"NN-contact-force/neuralnets/3D-5gon-PDD-R1");
            r2net.load(DART_DATA_PATH"NN-contact-force/neuralnets/3D-5gon-PDD-R2");
            r3net.load(DART_DATA_PATH"NN-contact-force/neuralnets/3D-5gon-PDD-R3");
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
    
    //==============================================================================
    bool isClose(const Eigen::Vector3d& pos1, const Eigen::Vector3d& pos2,
                 double tol)
    {
        return (pos1 - pos2).norm() < tol;
    }
    
    //==============================================================================
    void postProcess(dart::collision::CollisionResult& totalResult,
                     const dart::collision::CollisionResult& pairResult)
    {
        if (!pairResult.isCollision())
            return;
        
        // Don't add repeated points
        const auto tol = 1.0e-6;
        
        for (auto pairContact : pairResult.getContacts())
        {
            auto foundClose = false;
            
            for (auto totalContact : totalResult.getContacts())
            {
                if (isClose(pairContact.point, totalContact.point, tol))
                {
                    foundClose = true;
                    break;
                }
            }
            
            if (foundClose)
                continue;
            
            // do not add non-vertex points as well
            auto foundVertex = false;
            
            Eigen::MatrixXd vts(3,10);
            vts << -0.1500,  0.0500,   0.2000,   -0.0500,   -0.2000,   -0.1500,    0.0500,    0.2000,   -0.0500,   -0.2000,
            -0.1000,   -0.1000,   -0.1000,   -0.1000,   -0.1000,    0.1000,    0.1000,    0.1000,    0.1000,    0.1000,
            -0.1400,   -0.1400,    0.0100,    0.1100,   -0.0400,   -0.1400,   -0.1400,    0.0100,    0.1100,   -0.0400;
            
            for (int i = 0; i < 10; i++) {
                Eigen::Vector3d vt = vts.col(i);
                auto vt_pos = hand_bd->getWorldTransform()*(vt);
                
                if (isClose(pairContact.point, vt_pos, tol))
                {
                    foundVertex = true;
                    break;
                }
            }
            
            if (!foundVertex)
                continue;
            
            auto contact = pairContact;
            totalResult.addContact(contact);
        }
    }
    
    Eigen::Vector3d calculateCntFor3PointPentagon()
    {
        // see if it is top or bottom face
        bool topFace = true;
        bool bottomFace = true;
        for (auto contact : cleanResult.getContacts())  // should have 3 diff contacts
        {
            auto lP = hand_bd->getWorldTransform().inverse() * contact.point;
            if (std::abs(lP[1] - 0.1) > 1e-4)
                topFace = false;
            if (std::abs(lP[1] + 0.1) > 1e-4)
                bottomFace = false;
        }
        
        if (bottomFace)
            return hand_bd->getWorldTransform() * Eigen::Vector3d(-0.03,  -0.10, -0.04);
        if (topFace)
            return hand_bd->getWorldTransform() * Eigen::Vector3d(-0.03,  0.10, -0.04);
        
        // else: be side face
        Eigen::Vector3d dist((cleanResult.getContact(0).point-cleanResult.getContact(1).point).norm(),
                             (cleanResult.getContact(1).point-cleanResult.getContact(2).point).norm(),
                             (cleanResult.getContact(2).point-cleanResult.getContact(0).point).norm());
        if (dist.maxCoeff() == dist[0])
            return (cleanResult.getContact(0).point + cleanResult.getContact(1).point) / 2.0;
        else if (dist.maxCoeff() == dist[1])
            return (cleanResult.getContact(1).point + cleanResult.getContact(2).point) / 2.0;
        else
            return (cleanResult.getContact(2).point + cleanResult.getContact(0).point) / 2.0;
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
        dart::collision::CollisionResult result;
        result.clear();
        bool collision = collisionGroup->collide(option, &result);
        
        cleanResult.clear();
        postProcess(cleanResult, result);
        
        if (collision)
        {
            tempImp = Eigen::Vector6d::Zero();
            if (conModel == 0)
            {
                mWorld->getConstraintSolver()->getLastCollisionResult() = cleanResult;
                mWorld->getConstraintSolver()->solve();
            }
            if (conModel == 1)
            {
                myContactSolve(in_vec, cleanResult, vel_in_range);
            }
            
            if (conModel == 2)
            {
                pddContactSolve(in_vec, cleanResult, vel_in_range);
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

    double scaleInVec(Eigen::VectorXd& in_vec)
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
        if (std::abs(vel_uncons[4]) < 0.6)
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
    void myContactSolve(Eigen::VectorXd& in_vec, dart::collision::CollisionResult& result, bool vel_check)
    {
        if (!vel_check)
        {
            std::cout << "warning: PRE-CONTACT VEL OOR " 
                << hand_bd->getSpatialVelocity(Frame::World(),Frame::World()).transpose() << std::endl;
            mWorld->getConstraintSolver()->getLastCollisionResult() = result;
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
            double scale = scaleInVec(in_vec);
            
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
                result.clear();
                mWorld->getConstraintSolver()->getLastCollisionResult() = result;
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
                mWorld->getConstraintSolver()->getLastCollisionResult() = result;
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
                
                result.clear();
                mWorld->getConstraintSolver()->getLastCollisionResult() = result;
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
            double scale = scaleInVec(in_vec);
            
            vec_t inputnn;
            inputnn.assign(in_vec.data(), in_vec.data()+21);

            
            label_t c2output = c2net.predict_label(inputnn);
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
                
                result.clear();
                mWorld->getConstraintSolver()->getLastCollisionResult() = result;
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
                mWorld->getConstraintSolver()->getLastCollisionResult() = result;
                mWorld->getConstraintSolver()->solve(); // add normal impulse
                
                hand_bd -> setFrictionCoeff(1.0);
                ground_bd -> setFrictionCoeff(1.0);
            }
            else // c2out == 2
            {
                // do nothing, ignore collision solving
                mWorld->getConstraintSolver()->getCollisionGroup()->
                removeShapeFramesOf(ground_bd->getSkeleton().get());
                
                result.clear();
                mWorld->getConstraintSolver()->getLastCollisionResult() = result;
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
                cnt = calculateCntFor3PointPentagon();
            else // np = 4 or 5
            {
                for (auto contact : result.getContacts())
                {
                    cnt = cnt + contact.point;
                }
                cnt = cnt / result.getNumContacts();
            }
            
            auto cVel = hand_bd->getLinearVelocity(hand_bd->getWorldTransform().inverse() * cnt, Frame::World(), Frame::World());
            auto con_center = hand_bd->getCOM() - cnt;
            
            in_vec.segment<3>(15) = cVel;
            in_vec.segment<3>(18) = con_center;
            double scale = scaleInVec(in_vec);
            
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
                
                result.clear();
                mWorld->getConstraintSolver()->getLastCollisionResult() = result;
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
                mWorld->getConstraintSolver()->getLastCollisionResult() = result;
                mWorld->getConstraintSolver()->solve(); // add normal impulse
                
                hand_bd -> setFrictionCoeff(1.0);
                ground_bd -> setFrictionCoeff(1.0);
            }
            else // c3out == 2
            {
                // do nothing, ignore collision solving
                mWorld->getConstraintSolver()->getCollisionGroup()->
                removeShapeFramesOf(ground_bd->getSkeleton().get());
                
                result.clear();
                mWorld->getConstraintSolver()->getLastCollisionResult() = result;
                mWorld->getConstraintSolver()->solve();
                // restore
                mWorld->getConstraintSolver()->getCollisionGroup()->
                addShapeFramesOf(ground_bd->getSkeleton().get());
            }
        }
        else
        {
            std::cout << "WARNING: # of contact" << result.getNumContacts() << std::endl;
            mWorld->getConstraintSolver()->getLastCollisionResult() = result;
            mWorld->getConstraintSolver()->solve();
        }
    }
    
    // Yifeng: pass by reference: in_vec
    void pddContactSolve(Eigen::VectorXd& in_vec, dart::collision::CollisionResult& result, bool vel_check)
    {
        if (!vel_check)
        {
            std::cout << "warning: PRE-CONTACT VEL OOR "
            << hand_bd->getSpatialVelocity(Frame::World(),Frame::World()).transpose() << std::endl;
            mWorld->getConstraintSolver()->getLastCollisionResult() = result;
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
            double scale = scaleInVec(in_vec);
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
            double scale = scaleInVec(in_vec);
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
                cnt = calculateCntFor3PointPentagon();
            else // np = 4 or 5
            {
                for (auto contact : result.getContacts())
                {
                    cnt = cnt + contact.point;
                }
                cnt = cnt / result.getNumContacts();
            }
            
            auto cVel = hand_bd->getLinearVelocity(hand_bd->getWorldTransform().inverse() * cnt, Frame::World(), Frame::World());
            auto con_center = hand_bd->getCOM() - cnt;
            
            in_vec.segment<3>(15) = cVel;
            in_vec.segment<3>(18) = con_center;

            double scale = scaleInVec(in_vec);
            
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
        else
        {
            std::cout << "WARNING: # of contact" << result.getNumContacts() << std::endl;
            mWorld->getConstraintSolver()->getLastCollisionResult() = result;
            mWorld->getConstraintSolver()->solve();
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
    vector<Eigen::Vector3d> EndLinearPos;
    vector<Eigen::Quaterniond> EndAngularPos;
    vector<Eigen::Vector6d> FirstContactImpulse;
    Eigen::Vector6d tempImp;
    
    int conModel;
    
    dart::collision::CollisionResult cleanResult;
protected:
};


int main(int argc, char* argv[])
{
//    std::cout.precision(10);
    
    WorldPtr world = SkelParser::readWorld(DART_DATA_PATH"skel/mytest/pentagon_forTest.skel");
    assert(world != nullptr);
    world->setGravity(Eigen::Vector3d(0.0, -10.0, 0));
    MyWindow window(world);
    std::cout << "space bar: simulation on/off" << std::endl;
    glutInit(&argc, argv);
    window.initWindow(640, 480, "Simple Test");
    glutMainLoop();
}
