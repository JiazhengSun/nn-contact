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
//        std::cout << "t=" << ts ;
        if (ts != 0)
        {
            auto a = hand_bd->getSpatialVelocity(Frame::World(),Frame::World());
            theta_odo = theta_odo + a[2] * mWorld->getTimeStep();
//            std::cout << " " << theta_odo << std::endl;
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
        //        std::cout << in_vec.transpose() << std::endl;
        
        //        std::cout << vel_uncons.transpose() << std::endl;
        //        std::cout << pos[3] << " " << pos[4] << " " << theta_odo<< std::endl;
        
        auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
        auto collisionGroup = mWorld->getConstraintSolver()->getCollisionGroup();
        dart::collision::CollisionOption option;
        dart::collision::CollisionResult result;
        bool collision = collisionGroup->collide(option, &result);
        
        if (collision)
        {
            myContactSolve(in_vec, result);
            // mWorld->getConstraintSolver()->solve(); // grounf truth: DART contact solver
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
    
    void myContactSolve(Eigen::VectorXd in_vec, const dart::collision::CollisionResult& result)
    {
        Eigen::VectorXd in_c = in_vec;
        in_c[3] = in_c[3] / 20.0;
        in_c[4] = in_c[4] / 20.0;
        in_c[5] = in_c[5] / 20.0; // "normalize" input scale -> match the training data gathering
        
        vec_t input_c;
        input_c.assign(in_c.data(), in_c.data()+9);
        
        label_t c1output = c1net.predict_label(input_c);
        label_t c2output = c2net.predict_label(input_c);
        label_t c3output = c3net.predict_label(input_c);
        
        if (result.getNumContacts() == 1) // Point contact case
        {
            bool vel_check = ptCheckVelocityRange(in_vec);
            if (vel_check == true)
            {
                if (c1output == 1) // Static case -> apply constraints
                {
                    // set constraints
                    auto pos1 = result.getContact(0).point;
                    //auto pos2 = result.getContact(1).point;
                    auto constraint1 = std::make_shared<dart::constraint::BallJointConstraint>(hand_bd, ground_bd, pos1);
                    //auto constraint2 = std::make_shared<dart::constraint::BallJointConstraint>(hand_bd, ground_bd, pos2);
                    mWorld->getConstraintSolver()->addConstraint(constraint1);
                    //mWorld->getConstraintSolver()->addConstraint(constraint2);
                    
                    // disable ground from contact detection
                    mWorld->getConstraintSolver()->getCollisionGroup()->
                    removeShapeFramesOf(ground_bd->getSkeleton().get());
                    
                    // solve for impulses needed to maintain ball joints
                    mWorld->getConstraintSolver()->solve();

                    // restore
                    mWorld->getConstraintSolver()->removeConstraint(constraint1);
                    //mWorld->getConstraintSolver()->removeConstraint(constraint2);
                    mWorld->getConstraintSolver()->getCollisionGroup()->
                    addShapeFramesOf(ground_bd->getSkeleton().get());
                }
                else if (c1output == 0)
                {
                    // run regreesor
                    auto pos1 = result.getContact(0).point;
                    //auto pos2 = result.getContact(1).point;
                    
                    Eigen::VectorXd in_r = Eigen::VectorXd::Zero(9);
                    in_r = in_vec;
                    in_r[3] = in_r[3] / 20.0;
                    in_r[4] = in_r[4] / 20.0;
                    in_r[5] = in_r[5] / 20.0;
                    
                    // // TODO: a temporary fix utilizing linearity of the meta cone.
                    // Eigen::Vector3d velAbs;
                    // velAbs << in_r[3], in_r[4];
                    // velAbs = velAbs.cwiseAbs();
                    // double scale = 1.0;
                    // if (velAbs.maxCoeff() < 0.5 && std::abs(in_r[2] * (2.5/velAbs.maxCoeff()))<5.0)
                    // {
                    //     in_r[2] = in_r[2] * (2.5/velAbs.maxCoeff());
                    //     in_r[3] = in_r[3] * (2.5/velAbs.maxCoeff());
                    //     in_r[4] = in_r[4] * (2.5/velAbs.maxCoeff());
                    //     scale = 2.5/velAbs.maxCoeff();
                    // }
                    
                    vec_t input_r;
                    input_r.assign(in_r.data(), in_r.data()+9);
                    
                    vec_t r1output = r1net.predict(input_r);
                    // scaled down 100 times when training, should be okay if just train with large labels
                    double imp = *(r1output.begin()) * 100.0; //->match the /100 in during regressor training
                    Eigen::Vector3d result;
                    result << imp;
                    //result << px, pz, ptheta_y
                    // Eigen::Vector3d friction;
                    // friction << fric, 0.0, 0.0;
                    
                    // decouple instead of blindly run one step
                    hand_bd -> clearExternalForces();
                    mWorld->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
                    hand_bd -> addExtForce(friction, (pos1+pos2)/2.0, false, false);
                    hand_bd->getSkeleton()->computeForwardDynamics();
                    hand_bd->getSkeleton()->integrateVelocities(mWorld->getTimeStep());
                    
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
//                    double fric = *(r2output.begin()) * 100.0;
                    double fric = *(r2output.begin()) * 100.0 / scale;
                    
//                    std::cout << fric << std::endl;
                    Eigen::Vector3d friction;
                    friction << fric, 0.0, 0.0;

                    
                    // decouple instead of blindly run one step
                    hand_bd -> clearExternalForces();
                    mWorld->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
                    hand_bd -> addExtForce(friction, (pos1+pos3)/2.0, false, false);
                    hand_bd->getSkeleton()->computeForwardDynamics();
                    hand_bd->getSkeleton()->integrateVelocities(mWorld->getTimeStep());
                    
                    hand_bd -> setFrictionCoeff(0.0);
                    ground_bd -> setFrictionCoeff(0.0);
                    mWorld->getConstraintSolver()->solve();

                    //
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
    
    std::cout << "please input dx_0 and dtheta_0:" << std::endl;
    double x0, th0;
    std::cin >> x0 >> th0;
    
    // Create reference frames for setting the initial velocity
    Eigen::Isometry3d centerTf(Eigen::Isometry3d::Identity());
    centerTf.translation() = world->getSkeleton("hand skeleton")->getCOM();
    SimpleFrame center(Frame::World(), "center", centerTf);
    Eigen::Vector3d v = Eigen::Vector3d(x0, 0.0, 0.0);
    Eigen::Vector3d w = th0 * Eigen::Vector3d::UnitZ();
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
