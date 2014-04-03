#-------------------------------------------------
#
# Project created by QtCreator 2014-03-22T11:05:21
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = SquaresDetection
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp

INCLUDEPATH += D://opencv//sources//opencv_bin//install//include

LIBS += D://opencv//sources//opencv_bin//bin//*.dll
