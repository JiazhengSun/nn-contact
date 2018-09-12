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
        cout << "t: " << ts << endl;
        cout<<"Pos is: "<<pos.transpose()<<endl;
        vel_uncons = hand_bd->getSpatialVelocity(Frame::World(),Frame::World());
        cout<<"Velocity is: "<<vel_uncons.transpose()<<endl;
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
        
//        Eigen::Vector3d oldAng = hand_bd->getWorldTransform().linear() * hand_bd->getAngularMomentum();
        //cout<<result.getNumContacts()<<endl;
        
        if (collision)
        {
//            for (auto i : result.getContacts()) // access by value, the type of i is int
//                std::cout << i.point.transpose() << std::endl;
            
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
            cout << "Point Contact:" << c1output << endl;
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
                
                cout << scale << endl;
                
                cout<<"Point Contact Force is "<< fric_force.transpose()<<endl;
                
                hand_bd -> clearExternalForces();
                hand_bd -> addExtForce(fric_force, lP1, false, true);
                
                hand_bd -> addConstraintImpulse(hand_bd->getAspectState().mFext * mWorld->getTimeStep());
                std::cout << hand_bd -> getConstraintImpulse().transpose() << std::endl;
                hand_bd -> clearExternalForces();
                hand_bd -> getSkeleton()->computeImpulseForwardDynamics();
                
                hand_bd -> setFrictionCoeff(0.0);
                ground_bd -> setFrictionCoeff(0.0);
                
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
            cout << "Line Contact:" << c2output <<endl;
            
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
                
                cout<<"Line Contact Force/toque is "<< fric_force.transpose() << " t " << tauy << endl;
                
                hand_bd -> clearExternalForces();
                hand_bd -> addExtForce(fric_force, Eigen::Vector3d(0.0, 0.0, 0), false, true);
                hand_bd -> addExtTorque(torque, false);
                
                hand_bd -> addConstraintImpulse(hand_bd->getAspectState().mFext * mWorld->getTimeStep());
                hand_bd -> clearExternalForces();
                hand_bd -> getSkeleton()->computeImpulseForwardDynamics();
                
                hand_bd -> setFrictionCoeff(0.0);
                ground_bd -> setFrictionCoeff(0.0);
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
            cout << "Face Contact:" << c3output <<endl;
            
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
                
                cout<<"Face Contact Force/toque is "<< fric_force.transpose() << " t " << tauy << endl;
                
                hand_bd -> clearExternalForces();
                hand_bd -> addExtForce(fric_force, Eigen::Vector3d(0.0, 0.0, 0), false, true);
                hand_bd -> addExtTorque(torque, false);
                
                hand_bd -> addConstraintImpulse(hand_bd->getAspectState().mFext * mWorld->getTimeStep());
                hand_bd -> clearExternalForces();
                hand_bd -> getSkeleton()->computeImpulseForwardDynamics();
                
                hand_bd -> setFrictionCoeff(0.0);
                ground_bd -> setFrictionCoeff(0.0);
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
    
    int conModel;
    
    dart::collision::CollisionResult cleanResult;
protected:
};


int main(int argc, char* argv[])
{
    std::cout.precision(10);
    
    WorldPtr world = SkelParser::readWorld(DART_DATA_PATH"skel/mytest/pentagon_forTest.skel");
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
