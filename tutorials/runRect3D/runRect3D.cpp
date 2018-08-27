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

class MyWindow : public dart::gui::SimWindow
{
public:
    
    MyWindow(const WorldPtr& world)
    {
        setWorld(world);
        ts = 0;
        
        // Bob's trail
        if (UNSYMCONE)
        {
            c1net.load("rect_c1_unsym");
            c2net.load("rect_c2_unsym");
            c3net.load("rect_c3_unsym");
            r1net.load("rect_r1_unsym");
            r2net.load("rect_r2_unsym");
            r3net.load("rect_r3_unsym");
        } 
        else
        {
            c1net.load("rect_c1_sym");
            c2net.load("rect_c2_sym");
            c3net.load("rect_c3_sym");
            r1net.load("rect_r1_sym");
            r2net.load("rect_r2_sym");  
            r3net.load("rect_r3_sym");       
        }
        
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

        //BOBTODO: Change Neural net input!!
        Eigen::VectorXd in_vec = Eigen::VectorXd::Zero(9);
        in_vec << pos[0], pos[1], pos[2],
                  vel_uncons[0], vel_uncons[1], vel_uncons[2],
                  vel_uncons[3], vel_uncons[4], vel_uncons[5];
        
        auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
        auto collisionGroup = mWorld->getConstraintSolver()->getCollisionGroup();
        dart::collision::CollisionOption option;
        dart::collision::CollisionResult result;
        bool collision = collisionGroup->collide(option, &result);
        
        if (collision)
        {
            myContactSolve(in_vec, result);
            //mWorld->getConstraintSolver()->solve();
        }
        
        
        // Compute velocity changes given constraint impulses
        for (size_t i=0; i < mWorld->getNumSkeletons(); i++)
        {
            auto skel = mWorld->getSkeleton(i);
            if (!skel->isMobile())
                continue;
            
            if (skel->isImpulseApplied())
            {
                skel->computeImpulseForwardDynamics();
                skel->setImpulseApplied(false);
            }
            
            skel->integratePositions(mWorld->getTimeStep());
            skel->clearInternalForces();
            skel->clearExternalForces();
            skel->resetCommands();
        }
        ts++;
    }


    // make sure the range is within those in training data gathering in point case
    bool ptCheckVelocityRange(Eigen::VectorXd in_vec)
    {
        if (in_vec[3]<=20 && in_vec[3]>=-20 && in_vec[4]<=20 && in_vec[4]>=-20 && in_vec[5]<=20 && in_vec[5]>=-20) //angular
        {
            if (in_vec[6]<=5 && in_vec[6]>=-5 && in_vec[7]<=4 && in_vec[7]>=-6 && in_vec[8]<=5 && in_vec[8]>=-5) // linear
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

    bool checkVelocityRange(Eigen::VectorXd in_vec)
    {
        if (in_vec[3]<=20 && in_vec[3]>=-20 && in_vec[4]<=20 && in_vec[4]>=-20 && in_vec[5]<=20 && in_vec[5]>=-20) //angular
        {
            if (in_vec[6]<=2 && in_vec[6]>=-2 && in_vec[7]<=13 && in_vec[7]>=-13 && in_vec[8]<=2 && in_vec[8]>=-2) // linear
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
    
    void myContactSolve(Eigen::VectorXd in_vec, const dart::collision::CollisionResult& result)
    {
        double delTime = mWorld->getTimeStep();
        cout<<delTime<<endl;
        Eigen::VectorXd in_c = in_vec;
        in_c[3] = in_c[3] / 20.0;
        in_c[4] = in_c[4] / 20.0;
        in_c[5] = in_c[5] / 20.0; // "normalize" input scale -> match the training data gathering
        
        vec_t input_c;
        input_c.assign(in_c.data(), in_c.data()+9);
        
        label_t c1output = c1net.predict_label(input_c);
        label_t c2output = c2net.predict_label(input_c);
        label_t c3output = c3net.predict_label(input_c);

        // if (result.getNumContacts() != 0)
        // {
        //     cout<<"c1output is: "<<c1output<<endl;
        //     cout<<"c2output is: "<<c2output<<endl;
        //     cout<<"c2output is: "<<c3output<<endl;
        // }

        cout<<"Number of contacts are: "<<result.getNumContacts()<<endl;

        
        if (result.getNumContacts() == 1) // Point contact case
        {
            bool vel_check = ptCheckVelocityRange(in_vec);
            if (vel_check == true)
            {
                //cout<<"c1 output is: "<<c1output<<endl;
                if (c1output == 1) // Static case -> apply constraints
                {
                    // set constraints
                    auto pos = result.getContact(0).point;
                    auto constraint = std::make_shared<dart::constraint::BallJointConstraint>(hand_bd, ground_bd, pos);
                    mWorld->getConstraintSolver()->addConstraint(constraint);
                    
                    // disable ground from contact detection
                    mWorld->getConstraintSolver()->getCollisionGroup()->
                    removeShapeFramesOf(ground_bd->getSkeleton().get());
                    
                    // solve for impulses needed to maintain ball joints
                    mWorld->getConstraintSolver()->solve();

                    // restore
                    mWorld->getConstraintSolver()->removeConstraint(constraint);
                    //mWorld->getConstraintSolver()->removeConstraint(constraint2);
                    mWorld->getConstraintSolver()->getCollisionGroup()->
                    addShapeFramesOf(ground_bd->getSkeleton().get());
                }
                else if (c1output == 0)
                {
                    // run regreesor
                    // auto pos = result.getContact(0).point;
                    // //auto pos2 = result.getContact(1).point;
                    
                    // Eigen::VectorXd in_r = Eigen::VectorXd::Zero(9);
                    // in_r = in_vec;
                    // in_r[3] = in_r[3] / 20.0;
                    // in_r[4] = in_r[4] / 20.0;
                    // in_r[5] = in_r[5] / 20.0;
                    
                    // vec_t input_r;
                    // input_r.assign(in_r.data(), in_r.data()+9);
                    
                    // vec_t r1output = r1net.predict(input_r);
                    // // * 100 to match the /100 during regression training
                    // double fx = r1output[0] * 100.0 / delTime;
                    // double fz = r1output[1] * 100.0 / delTime;
                    // Eigen::Vector3d friction;
                    // friction << fx, 0.0, fz;
                    
                    // // decouple instead of blindly run one step
                    // // applying the impulse. Need modificaiton!
                    // hand_bd -> clearExternalForces();
                    // mWorld->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
                    // hand_bd -> addExtForce(friction, pos, false, false);
                    // hand_bd->getSkeleton()->computeForwardDynamics();
                    // hand_bd->getSkeleton()->integrateVelocities(mWorld->getTimeStep());
                    
                    // hand_bd -> setFrictionCoeff(0.0);
                    // ground_bd -> setFrictionCoeff(0.0);
                    mWorld->getConstraintSolver()->solve();

                    // // restore
                    // hand_bd -> setFrictionCoeff(1.0);
                    // ground_bd ->setFrictionCoeff(1.0);
                    // mWorld->setGravity(gravity);
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
            else
            {
                std::cout << "warning: c1 OOR " << in_vec.transpose() << std::endl;
                mWorld->getConstraintSolver()->solve();
            }
        }
        else if (result.getNumContacts() == 2) // Line contact
        {
            bool vel_check = checkVelocityRange(in_vec);
            if (vel_check == true)
            {
                //cout<<"c2 output is: "<<c2output<<endl;
                if (c2output == 1)
                {
                    auto pos1 = result.getContact(0).point;
                    auto pos2 = result.getContact(1).point;
                    auto constraint1 = std::make_shared<dart::constraint::BallJointConstraint>(hand_bd, ground_bd, pos1);
                    auto constraint2 = std::make_shared<dart::constraint::BallJointConstraint>(hand_bd, ground_bd, pos2);
                    
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
                    // run regreesor
//                     auto pos1 = result.getContact(0).point;
//                     auto pos2 = result.getContact(1).point;
                    
//                     Eigen::VectorXd in_r = Eigen::VectorXd::Zero(9);
//                     in_r = in_vec;
//                     in_r[3] = in_r[3] / 20.0;
//                     in_r[4] = in_r[4] / 20.0;
//                     in_r[5] = in_r[5] / 20.0;
                    
//                     vec_t input_r;
//                     input_r.assign(in_r.data(), in_r.data()+9);
                    
//                     vec_t r2output = r2net.predict(input_r);
//                     // scaled down 100 times when training
//                     double fx = r2output[0] * 100.0 /delTime;
//                     double fz = r2output[1] * 100.0 /delTime;
//                     double ty = r2output[2] * 100.0 /delTime;
// //                    std::cout << fric << std::endl;
//                     Eigen::Vector3d friction;
//                     friction << fx, 0.0, fz;

//                     Eigen::Vector3d torque;
//                     torque << 0.0, ty, 0.0;
//                     // decouple instead of blindly run one step
//                     hand_bd -> clearExternalForces();
//                     mWorld->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
//                     // middle of the point
//                     hand_bd -> addExtForce(friction, (pos1+pos2)/2, false, false);
//                     hand_bd -> addExtTorque(torque, false);
//                     hand_bd->getSkeleton()->computeForwardDynamics();
//                     hand_bd->getSkeleton()->integrateVelocities(mWorld->getTimeStep());
                    
//                     hand_bd -> setFrictionCoeff(0.0);
//                     ground_bd -> setFrictionCoeff(0.0);
                     mWorld->getConstraintSolver()->solve();

//                     hand_bd -> setFrictionCoeff(1.0);
//                     ground_bd ->setFrictionCoeff(1.0);
//                     mWorld->setGravity(gravity);
                }
                else // c2out == 2
                {
                    mWorld->getConstraintSolver()->getCollisionGroup()->
                    removeShapeFramesOf(ground_bd->getSkeleton().get());       
                    mWorld->getConstraintSolver()->solve();
                    mWorld->getConstraintSolver()->getCollisionGroup()->
                    addShapeFramesOf(ground_bd->getSkeleton().get());
                }
            }
            else
            {
                std::cout << "warning: c2 OOR " << in_vec.transpose() << std::endl;
                mWorld->getConstraintSolver()->solve();
            }
        }
        else if (result.getNumContacts() == 4) // surface contact
        {
            bool vel_check = checkVelocityRange(in_vec);
            if (vel_check == true)
            {
                //cout<<"c3 output is: "<<c3output<<endl;
                if (c3output == 1)
                {
                    auto pos1 = result.getContact(0).point;
                    auto pos2 = result.getContact(1).point;
                    auto pos3 = result.getContact(2).point;
                    auto pos4 = result.getContact(3).point;

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
                    Eigen::Vector6d old_vel = hand_bd->getSpatialVelocity(Frame::World(),Frame::World());
                    cout<<"Old velocities are: "<<old_vel.transpose()<<endl;
                    // run regreesor
                    auto pos1 = result.getContact(0).point;
                    auto pos2 = result.getContact(1).point;
                    auto pos3 = result.getContact(2).point;
                    auto pos4 = result.getContact(3).point;
                    
                    Eigen::VectorXd in_r = Eigen::VectorXd::Zero(9);
                    in_r = in_vec;
                    in_r[3] = in_r[3] / 20.0;
                    in_r[4] = in_r[4] / 20.0;
                    in_r[5] = in_r[5] / 20.0;
                    
                    vec_t input_r;
                    input_r.assign(in_r.data(), in_r.data()+9);
                    
                    vec_t r3output = r3net.predict(input_r);
                    // scaled down 100 times when training
                    double fx = r3output[0] * 100.0 /delTime;
                    double fz = r3output[1] * 100.0 /delTime;
                    double ty = r3output[2] * 100.0 /delTime;

                    double tempX = 19.406 /delTime;
                    double tempZ = -242.082 /delTime;
                    double tempTY = -4.09366 /delTime;

                    Eigen::Vector3d friction;
                    friction << fx, 0.0, fz;
                    //friction << tempX, 0.0, tempZ;
                    Eigen::Vector3d torque;
                    torque << 0.0, ty, 0.0;
                    //torque << 0.0, tempTY, 0.0;
                    // decouple instead of blindly run one step
                    hand_bd -> clearExternalForces();
                    mWorld->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
                    // middle of the point
                    hand_bd -> addExtForce(friction, (pos1+pos2+pos3+pos4)/4, false, false);
                    hand_bd -> addExtTorque(torque, false);
                    hand_bd->getSkeleton()->computeForwardDynamics();
                    hand_bd->getSkeleton()->integrateVelocities(mWorld->getTimeStep());
                    
                    hand_bd -> setFrictionCoeff(0.0);
                    ground_bd -> setFrictionCoeff(0.0);
                    mWorld->getConstraintSolver()->solve();
                    
                    Eigen::Vector6d new_vel = hand_bd->getSpatialVelocity(Frame::World(),Frame::World());
                    cout<<"New velocities are: "<<new_vel.transpose()<<endl;
                    // restore
                    hand_bd -> setFrictionCoeff(1.0);
                    ground_bd ->setFrictionCoeff(1.0);
                    mWorld->setGravity(gravity);
                }
                else // c3output = 2
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
                std::cout << "warning: c3 OOR " << in_vec.transpose() << std::endl;
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
    network<sequential> c3net;
    network<sequential> r1net;
    network<sequential> r2net;
    network<sequential> r3net;
    
    Eigen::Vector3d gravity;
    
    BodyNodePtr hand_bd;
    BodyNodePtr ground_bd;
    
protected:
};


int main(int argc, char* argv[])
{
    WorldPtr world = SkelParser::readWorld(DART_DATA_PATH"/NN-contact-force/skel/singleBody_test.skel");
    assert(world != nullptr);
    
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
