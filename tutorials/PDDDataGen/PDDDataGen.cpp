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
#define NUMSAM 50

double StartPos[] = {}; //vector 6d.

double StartVel[] = {}; //vector 6d

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
