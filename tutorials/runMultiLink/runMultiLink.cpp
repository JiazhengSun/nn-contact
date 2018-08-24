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

// Eigen should have been installed since DART depends on it
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include "tiny_dnn/tiny_dnn.h"

using namespace dart::dynamics;
using namespace dart::constraint;
using namespace dart::simulation;
using namespace dart;
using namespace utils;
using namespace Eigen;

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

// ***********
// we assume we have no external forces rather than gravity.
// otherwise code could be more tedious (Not sure)

// *********** Eigen equation solver (minpack) err msgs:
//errors = {0: "Improper input parameters were entered.",
//    1: "The solution converged.",
//    2: "The number of calls to function has "
//    "reached maxfev = %d." % maxfev,
//    3: "xtol=%f is too small, no further improvement "
//    "in the approximate\n  solution "
//    "is possible." % xtol,
//    4: "The iteration is not making good progress, as measured "
//    "by the \n  improvement from the last five "
//    "Jacobian evaluations.",
//    5: "The iteration is not making good progress, "
//    "as measured by the \n  improvement from the last "
//    "ten iterations.",
//    'unknown': "An error occurred."}


class MyWindow : public dart::gui::SimWindow
{
public:
    
    MyWindow(const WorldPtr& world) : func(*this)
    {
        setWorld(world);
        
        hand_bd = mWorld->getSkeleton("hand skeleton")->getBodyNode("hand");
        bicep_bd = mWorld->getSkeleton("arm skeleton")->getBodyNode("bicep");
        ground_bd = mWorld->getSkeleton("ground skeleton")->getBodyNode("ground");
        
        // this is a "virtual" ball joint where we do not use DART LCP to solve its forces
        // instead, we use the Eigen nonlinear solver to solve the joint forces (x)
        // we construct a ball joint here to gather infomation and to build the nonlinear equation G(x)=0
        // ****** the position of the ball joint
        Eigen::Vector3d pos1(0.8, 0.0, 0.0);
        
        ball_constraint1 = std::make_shared<dart::constraint::BallJointConstraint>(hand_bd, bicep_bd, pos1);
        mWorld->getConstraintSolver()->addConstraint(ball_constraint1);
        
        mOffset1 = hand_bd->getTransform().inverse() * pos1;
        mOffset2 = bicep_bd->getTransform().inverse() * pos1;
        
        c1net.load(DART_DATA_PATH"/NN-contact-force/neuralnets/net-c1-feb24");
        c2net.load(DART_DATA_PATH"/NN-contact-force/neuralnets/net-c2-feb25");
        r1net.load(DART_DATA_PATH"/NN-contact-force/neuralnets/net-r1-feb26");
        r2net.load(DART_DATA_PATH"/NN-contact-force/neuralnets/net-r2-feb25");
        
        ts = 0;
        
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
        // And change lighting position
        GLfloat light_position[] = { 0.0, 5.0, 5.0, 0.0 };
        glLightfv(GL_LIGHT0, GL_POSITION, light_position);
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        SimWindow::drawWorld();
    }
    
    // time stepping called by GUI
    void timeStepping() override
    {
        // Integrate velocity for unconstrained skeletons (integrate g)
        for (size_t i=0; i < mWorld->getNumSkeletons(); i++)
        {
            auto skel = mWorld->getSkeleton(i);
            if (!skel->isMobile())
                continue;
            
            skel->computeForwardDynamics();
            skel->integrateVelocities(mWorld->getTimeStep());
        }
        
        // store state vectors because solver runs many iterations; reset before each iteration
        pos_m = bicep_bd->getSkeleton()->getPositions();
        vel_m = bicep_bd->getSkeleton()->getVelocities();
        
        pos_s = hand_bd->getSkeleton()->getPositions();
        vel_s = hand_bd->getSkeleton()->getVelocities();
        
        // Obtain initial guess for G(x) = 0 by calculating A^{-1}c_0
        ball_constraint1->update();
        
        auto ballJointPtr = dynamic_cast<BallJointConstraint*>(ball_constraint1.get());
        c_0 = ballJointPtr->mJacobian2 * bicep_bd->getSpatialVelocity();
        
        ball_constraint1->excite();
    
        for (int j=0; j<3; j++)
        {
            ball_constraint1->applyUnitImpulse(j);
            ball_constraint1->getVelocityChange(A.data()+3*j, true);
        }
        
        ball_constraint1->unexcite();
        hand_bd->getSkeleton()->clearConstraintImpulses();
        bicep_bd->getSkeleton()->clearConstraintImpulses();
        
        solveWithHybrd1();      // the solution reached at final iteration will be used as next state
        // so simulation implicitly preceeds
        
        // DART ground truth
//        mWorld->getConstraintSolver()->solve();
//        // Compute velocity changes given constraint impulses
//        for (size_t i=0; i < mWorld->getNumSkeletons(); i++)
//        {
//            auto skel = mWorld->getSkeleton(i);
//            if (!skel->isMobile())
//                continue;
//            
//            if (skel->isImpulseApplied())
//            {
//                skel->computeImpulseForwardDynamics();
//                skel->setImpulseApplied(false);
//            }
//            
//            skel->integratePositions(mWorld->getTimeStep());
//            skel->clearInternalForces();
//            skel->clearExternalForces();
//            skel->resetCommands();
//        }
        
        ts++;
    }
    
    // Generic functor
    template<typename _Scalar, int NX=Dynamic, int NY=Dynamic>
    struct Functor
    {
        typedef _Scalar Scalar;
        enum {
            InputsAtCompileTime = NX,
            ValuesAtCompileTime = NY
        };
        typedef Matrix<Scalar,InputsAtCompileTime,1> InputType;
        typedef Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
        typedef Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;
    
        const int m_inputs, m_values;
    
        Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
        Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}
    
        int inputs() const { return m_inputs; }
        int values() const { return m_values; }
    
        // you should define that in the subclass :
        //  void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
    };
    
    // forward evaluting of fvec = G(x); solver complete when fvec == 0;
    struct my_functor : Functor<double>
    {
        my_functor(MyWindow& window) : Functor<double>(2,2), _parent(window) {}
        int operator()(const VectorXd &x, VectorXd &fvec) const
        {
            //  G(x) is the vel difference between 2 joint points
            // i.e. the joint force x needs to maintain joint
            
            // match pos and vel states (after integrating g)
            _parent.bicep_bd->getSkeleton()->setPositions(_parent.pos_m);
            _parent.bicep_bd->getSkeleton()->setVelocities(_parent.vel_m);
            
            _parent.hand_bd->getSkeleton()->setPositions(_parent.pos_s);
            _parent.hand_bd->getSkeleton()->setVelocities(_parent.vel_s);
            
            // integrate current f_jcon =: x
            Eigen::Vector3d x3d(x[0], x[1], 0.0);
            auto invTimestep = 1.0 / _parent.mWorld->getTimeStep();
            auto ballJointPtr = dynamic_cast<BallJointConstraint*>(_parent.ball_constraint1.get());
            _parent.ball_constraint1->update();
            auto state1 = dart::dynamics::detail::BodyNodeState(ballJointPtr->mJacobian1.transpose() * x3d * invTimestep);
            _parent.hand_bd->setAspectState(state1);        // just means set external force, but no API to set 6D wrench..
            auto state2 = dart::dynamics::detail::BodyNodeState(ballJointPtr->mJacobian2.transpose() * x3d * -invTimestep);
            _parent.bicep_bd->setAspectState(state2);
            
            // remove ball joint
            _parent.mWorld->getConstraintSolver()->removeConstraint(_parent.ball_constraint1);
            
            // do not integrate g again
            _parent.mWorld->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
            
            // Integrate velocity for unconstrained skeletons
            // position is not changed
            for (size_t i=0; i < _parent.mWorld->getNumSkeletons(); i++)
            {
                auto skel = _parent.mWorld->getSkeleton(i);
                if (!skel->isMobile())
                    continue;
                
                skel->computeForwardDynamics();
                skel->integrateVelocities(_parent.mWorld->getTimeStep());
            }
            
            // restore g
            _parent.mWorld->setGravity(_parent.gravity);
            
            
            auto vel_uncons = _parent.hand_bd->getSpatialVelocity(Frame::World(),Frame::World());
            Eigen::VectorXd in_vec = Eigen::VectorXd::Zero(5);
            in_vec << sin(_parent.pos_s[2]), cos(_parent.pos_s[2]), vel_uncons[2], vel_uncons[3], vel_uncons[4];
            
            auto collisionEngine = _parent.mWorld->getConstraintSolver()->getCollisionDetector();
            auto collisionGroup = _parent.mWorld->getConstraintSolver()->getCollisionGroup();
            dart::collision::CollisionOption option;
            dart::collision::CollisionResult result;
            bool collision = collisionGroup->collide(option, &result);
            
            if (collision)
            {
                std::cout << result.getNumContacts() << std::endl;
                // constraint solver and pos intergrate.
                _parent.myContactSolve(in_vec, result);
                
//                _parent.mWorld->getConstraintSolver()->solve();
            }

            // Compute velocity changes given constraint impulses
            for (size_t i=0; i < _parent.mWorld->getNumSkeletons(); i++)
            {
                auto skel = _parent.mWorld->getSkeleton(i);
                if (!skel->isMobile())
                    continue;
                
                if (skel->isImpulseApplied())
                {
                    skel->computeImpulseForwardDynamics();
                    skel->setImpulseApplied(false);
                }
                
                skel->integratePositions(_parent.mWorld->getTimeStep());
                skel->clearInternalForces();
                skel->clearExternalForces();
                skel->resetCommands();
            }
            
            // Evaluate G(x), the delta vel.
            Eigen::Vector3d G = -ballJointPtr->mJacobian1 * _parent.hand_bd->getSpatialVelocity();
            G += ballJointPtr->mJacobian2 * _parent.bicep_bd->getSpatialVelocity();
            
            // A better way of evaluating G(x) (maybe):
            // In evaluating vel residue, impulse-based solver might be (?) better with
            // feedback error reduction. See getInformation() for detials.
            // If we doing so, we should use old pos (and new vel). Store new pos somewhere else and restore later.
//            _parent.newpos_m = _parent.mWorld->getSkeleton("arm skeleton")->getPositions();
//            _parent.newpos_s = _parent.mWorld->getSkeleton("hand skeleton")->getPositions();
//            _parent.mWorld->getSkeleton("arm skeleton")->setPositions(_parent.pos_m);
//            _parent.mWorld->getSkeleton("hand skeleton")->setPositions(_parent.pos_s);
//            
//            // Evaluate G(x), the delta vel.
//            _parent.ball_constraint1->update();
//            double* G = new double[3];
//            
//            double* xtmp = new double[3];
//            double* w = new double[3];
//            double* lo = new double[3];
//            double* hi = new double[3];
//            int* findex = new int[3];
//            ConstraintInfo constInfo;
//            constInfo.x      = xtmp;
//            constInfo.lo     = lo;
//            constInfo.hi     = hi;
//            constInfo.b      = G;       // only care about this; all other vars are temporary
//            constInfo.findex = findex;
//            constInfo.w      = w;
//            constInfo.invTimeStep = 1.0 / _parent.mWorld->getTimeStep(); // emmm.......
//            _parent.ball_constraint1->getInformation(&constInfo);
//            delete[] xtmp; // I only need b, delete all the others.
//            delete[] w;
//            delete[] lo;
//            delete[] hi;
//            delete[] findex;
            
//            std::cout << x.transpose() << std::endl;
            fvec[0] = G[0];
            fvec[1] = G[1];
            // g[2] == 0;
//            std::cout << "g" << g[0] << " " << g[1] << std::endl;
            
            Eigen::Vector2d gtmp(G[0], G[1]);   // stop iteration when error small enough
            if (gtmp.norm() < 1e-5)
            {
                fvec[0] = 0.0;
                fvec[1] = 0.0;
            }
            
            // restore ball joint
            assert(_parent.ball_constraint1);
            _parent.mWorld->getConstraintSolver()->addConstraint(_parent.ball_constraint1);
            
//            // restore new pos states *******If using error feedback
//            _parent.mWorld->getSkeleton("arm skeleton")->setPositions(_parent.newpos_m);
//            _parent.mWorld->getSkeleton("hand skeleton")->setPositions(_parent.newpos_s);
            
            // restore Fext states
            _parent.bicep_bd->getSkeleton()->clearExternalForces();
            _parent.hand_bd->getSkeleton()->clearExternalForces();
            
            return 0;
        }
        MyWindow& _parent;
    };
    
    void solveWithHybrd1()
    {
        int n=2, info;
        VectorXd x(n);
        
        Eigen::Vector3d x3d_0 = A.colPivHouseholderQr().solve(c_0);
        
        /* the following starting values provide a rough solution. */
        x[0] = x3d_0[0];
        x[1] = x3d_0[1];
        if (abs(x3d_0[2]) > 1e-4)
            std::cout << "Warn: x3d[2]:" << x3d_0[2] << std::endl;
        
        // do the computation
        HybridNonLinearSolver<my_functor> solver(func);
        
        info = solver.hybrd1(x, 5e-3);
        
        // check return value
//        std::cout << "s" << info << std::endl;
//        std::cout << solver.nfev << std::endl;
//        std::cout << "solution:" << x.transpose() << std::endl << std::endl;
    }
    
    // same as the single body solver
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
                    
                    //                    Eigen::Vector3d newVel = hand_bd->getLinearVelocity(Frame::World(),Frame::World());
                    //                    auto normal = (newVel[1] - in_vec[4])/0.002;
                    //                    if (normal >= 0)
                    //                    {
                    //                        std::cout << "ggg" << std::endl;
                    //                    }
                    //                    else
                    //                    {
                    //                        std::cout << "bbb" << std::endl;
                    //                    }
                    
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
                    velAbs << in_r[3], in_r[4];
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
                    //                    double fric = *(r1output.begin()) * 100.0;
                    //                    std::cout << fric << std::endl;
                    Eigen::Vector3d friction;
                    friction << fric, 0.0, 0.0;
                    
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
                    // set 1 weld constraints
                    //                    std::cout << "alov1" << std::endl;
                    
                    auto constraint = std::make_shared<dart::constraint::WeldJointConstraint>(hand_bd, ground_bd);
                    
                    mWorld->getConstraintSolver()->addConstraint(constraint);
                    
                    // disable ground from contact detection
                    mWorld->getConstraintSolver()->getCollisionGroup()->
                    removeShapeFramesOf(ground_bd->getSkeleton().get());
                    
                    mWorld->getConstraintSolver()->solve();
                    
                    //                    Eigen::Vector3d newVel = hand_bd->getLinearVelocity(Frame::World(),Frame::World());
                    //                    auto normal = (newVel[1] - in_vec[4])/0.002;
                    //                    if (normal >= 0)
                    //                    {
                    //                        std::cout << "ggg" << std::endl;
                    //                    }
                    //                    else
                    //                    {
                    //                        std::cout << "bbb" << std::endl;
                    //                    }
                    
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
    my_functor func;
    
    BodyNodePtr hand_bd;
    BodyNodePtr bicep_bd;
    BodyNodePtr ground_bd;
    ConstraintBasePtr ball_constraint1;
    
    Eigen::Vector3d c_0; // c for initial guess
    Eigen::Matrix3d A;
    
    Eigen::VectorXd pos_m;
    Eigen::VectorXd vel_m;
    Eigen::VectorXd pos_s;
    Eigen::VectorXd vel_s;
    
    Eigen::VectorXd newpos_m;
    Eigen::VectorXd newvel_m;
    Eigen::VectorXd newpos_s;
    Eigen::VectorXd newvel_s;
    
    network<sequential> c1net;
    network<sequential> c2net;
    network<sequential> r1net;
    network<sequential> r2net;
    
    Eigen::Vector3d gravity;
    
    Eigen::Vector3d mOffset1;
    Eigen::Vector3d mOffset2;
};

int main(int argc, char* argv[])
{
    WorldPtr world = SkelParser::readWorld(DART_DATA_PATH"/NN-contact-force/skel/multiLink_test.skel");
    assert(world != nullptr);
    
//    world->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
//    world->setTimeStep(0.002);
    world->getConstraintSolver()->setCollisionDetector(dart::collision::OdeCollisionDetector::create());
    
    
//    world->getSkeleton("arm")->getJoint("j_bicep_left")->setPositionLimitEnforced(true);
//    world->getSkeleton("arm")->getJoint("j_forearm_left")->setPositionLimitEnforced(true);
    
//    world->getSkeleton("arm")->enableSelfCollisionCheck();
//    world->getSkeleton("arm")->disableAdjacentBodyCheck();
    
    MyWindow window(world);
    
    std::cout << "space bar: simulation on/off" << std::endl;
    
    glutInit(&argc, argv);
    window.initWindow(640, 480, "Simple Test");
    glutMainLoop();
}

