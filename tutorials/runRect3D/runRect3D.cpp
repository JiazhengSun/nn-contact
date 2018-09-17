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
//cout << "Point local coordinates: "<<hand_bd->getWorldTransform().inverse() * PP1<<endl;
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
        if (ts == 800){cout<<"DONEEEEEEEEEEEEEDONEDONEDONE"<<endl;}
        if (collision)
        {
            int numCon = result.getNumContacts();
            if(numCon == 1)
            {
                auto P1 = result.getContact(0).point;
                auto LocalP1 = hand_bd->getWorldTransform().inverse() * P1;
                if((LocalP1-P1_h).norm() >= 1e-2)
                {
                    cout<<"Point Contact!"<<endl;
                    cout<<"Contact position is: "<<(hand_bd->getWorldTransform().inverse() * P1).transpose()<<endl;
                    P1_h = LocalP1;
                }

            }
            if(numCon == 2)
            {
                auto P1 = result.getContact(0).point;
                auto LocalP1 = hand_bd->getWorldTransform().inverse() * P1;
                auto P2 = result.getContact(1).point;
                auto LocalP2 = hand_bd->getWorldTransform().inverse() * P2;
                if((LocalP1 - L1).norm() >= 1e-2 && (LocalP2 - L2).norm() >= 1e-2)
                {
                    cout<<"Line Contact!"<<endl;
                    cout<<"Contact Pt1 is: "<<LocalP1.transpose()<<endl;
                    cout<<"Contact Pt2 is: "<<LocalP2.transpose()<<endl;
                    L1 = LocalP1; L2 = LocalP2;
                }

            }
            if(numCon >=3)
            {
                auto P1 = result.getContact(0).point;
                auto LocalP1 = hand_bd->getWorldTransform().inverse() * P1;
                auto P2 = result.getContact(1).point;
                auto LocalP2 = hand_bd->getWorldTransform().inverse() * P2;
                auto P3 = result.getContact(2).point;
                auto LocalP3 = hand_bd->getWorldTransform().inverse() * P3;
                if((LocalP1 - F1).norm() >= 1e-2 && (LocalP2 - F2).norm() >= 1e-2 && (LocalP3 - F3).norm() >= 1e-2)
                {
                    cout<<"Face Contact!"<<endl;
                    cout<<"Contact Pt1 is: "<<LocalP1.transpose()<<endl;
                    cout<<"Contact Pt2 is: "<<LocalP2.transpose()<<endl;
                    cout<<"Contact Pt3 is: "<<LocalP3.transpose()<<endl;
                    F1 = LocalP1; F2 = LocalP2; F3 = LocalP3;
                }
            }
            
            
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
    
    Eigen::Vector3d PrecisionLimit(Eigen::Vector3d input)
    {
        double vx = input[0]; double vy = input[1]; double vz = input[2];
        double scale = 0.01;  // i.e. round to nearest one-hundreth
        double new_x = (int)(vx / scale) * scale;
        double new_y = (int)(vy / scale) * scale;
        double new_z = (int)(vz / scale) * scale;
        return Eigen::Vector3d(new_x, new_y, new_z);
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
                
                hand_bd -> clearExternalForces();
                hand_bd -> addExtForce(fric_force, lP1, false, true);
                
                hand_bd -> addConstraintImpulse(hand_bd->getAspectState().mFext * mWorld->getTimeStep());
//                std::cout << hand_bd -> getConstraintImpulse().transpose() << std::endl;
                hand_bd -> clearExternalForces();
                hand_bd -> getSkeleton()->computeImpulseForwardDynamics();
                
                hand_bd -> setFrictionCoeff(0.0);
                ground_bd -> setFrictionCoeff(0.0);
                
                hand_bd -> setConstraintImpulse(Eigen::Vector6d::Zero());
                mWorld->getConstraintSolver()->solve();
                
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
                
                hand_bd -> clearExternalForces();
                hand_bd -> addExtForce(fric_force, Eigen::Vector3d(0.0, 0.0, 0), false, true);
                hand_bd -> addExtTorque(torque, false);
                
                hand_bd -> addConstraintImpulse(hand_bd->getAspectState().mFext * mWorld->getTimeStep());
                hand_bd -> clearExternalForces();
                hand_bd -> getSkeleton()->computeImpulseForwardDynamics();
                
                hand_bd -> setFrictionCoeff(0.0);
                ground_bd -> setFrictionCoeff(0.0);
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
                
                hand_bd -> clearExternalForces();
                hand_bd -> addExtForce(fric_force, Eigen::Vector3d(0.0, 0.0, 0), false, true);
                hand_bd -> addExtTorque(torque, false);
                
                hand_bd -> addConstraintImpulse(hand_bd->getAspectState().mFext * mWorld->getTimeStep());
                hand_bd -> clearExternalForces();
                hand_bd -> getSkeleton()->computeImpulseForwardDynamics();
                
                hand_bd -> setFrictionCoeff(0.0);
                ground_bd -> setFrictionCoeff(0.0);
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
            Eigen::Vector3d con_torque;
            con_torque << tx, ty, tz;
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
    
    Eigen::Vector3d P1_h= Eigen::Vector3d::Zero();
    Eigen::Vector3d L1 = Eigen::Vector3d::Zero(); Eigen::Vector3d L2 = Eigen::Vector3d::Zero();
    Eigen::Vector3d F1 = Eigen::Vector3d::Zero(); Eigen::Vector3d F2 = Eigen::Vector3d::Zero();
    Eigen::Vector3d F3 = Eigen::Vector3d::Zero();
    
    int conModel;
    
    dart::collision::CollisionResult result;
protected:
};


int main(int argc, char* argv[])
{
    std::cout.precision(10);
    
    WorldPtr world = SkelParser::readWorld(DART_DATA_PATH"/skel/mytest/hopperfoot2d_forTest.skel");
    assert(world != nullptr);
    
    world->setGravity(Eigen::Vector3d(0.0, -10.0, 0));
    
    std::cout << "please input dtheta_x0, dtheta_y0, dtheta_z0 and dx_0, dy_0, dz_0:" << std::endl;
    double x0, y0, z0, theta_x, theta_y, theta_z;
    std::cin >> theta_x >> theta_y >> theta_z >> x0 >> y0 >> z0;
    
    // Create reference frames for setting the initial velocity
    Eigen::Isometry3d centerTf(Eigen::Isometry3d::Identity());
    centerTf.translation() = world->getSkeleton("hand skeleton")->getCOM();
    SimpleFrame center(Frame::World(), "center", centerTf);
    Eigen::Vector3d v = Eigen::Vector3d(x0, y0, z0);
    Eigen::Vector3d w = Eigen::Vector3d(theta_x, theta_y, theta_z);
    center.setClassicDerivatives(v, w);
    SimpleFrame ref(&center, "root_reference");
    // ?
    ref.setRelativeTransform(world->getSkeleton("hand skeleton")->getBodyNode(0)->getTransform(&center));
    world->getSkeleton("hand skeleton")->getJoint(0)->setVelocities(ref.getSpatialVelocity());
    
    MyWindow window(world);
    
    std::cout << "space bar: simulation on/off" << std::endl;
    
    glutInit(&argc, argv);
    window.initWindow(640, 480, "Simple Test");
    
    glutMainLoop();
}
