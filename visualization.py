#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import sys
import math
import numpy
import cProfile
import pstats
import geneticvehicle

from PyQt4 import QtCore, QtGui, QtOpenGL
from PyQt4.QtCore import Qt
from functools import partial
from geneticvehicle import GeneticVehicle
from geneticvehicle import grouper


class Circle(QtGui.QGraphicsItem):

    def __init__(self, radius=1, parent=None):
        QtGui.QGraphicsItem.__init__(self, parent)
        self.radius = radius

    def paint(self, painter, style, widget=None):
        gradient = QtGui.QLinearGradient(0, 0, self.radius, self.radius)
        gradient.setColorAt(0, Qt.white)
        gradient.setColorAt(1, Qt.darkGray)
        painter.setBrush(QtGui.QBrush(gradient))
        painter.drawEllipse(QtCore.QPointF(0,0), self.radius, self.radius)
        painter.setPen(Qt.red)
        painter.drawLine(QtCore.QPointF(0, self.radius),QtCore.QPointF(0, -self.radius))

    def boundingRect(self):
        return self.shape().boundingRect()

    def shape(self):
        path = QtGui.QPainterPath()
        path.addEllipse(QtCore.QPointF(0, 0), self.radius, self.radius)
        return path


class Car(QtGui.QGraphicsItem):

    red = QtGui.QColor(255, 142, 142)

    def __init__(self, center, vertices, bounding_radius, vertex_colors, parent=None):
        QtGui.QGraphicsItem.__init__(self, parent)
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
        self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
        self.setCacheMode(QtGui.QGraphicsItem.DeviceCoordinateCache)
        self.setPos(center[0], center[1])
        self.vertices = vertices
        self.points = map(lambda vertex : QtCore.QPointF(*vertex), self.vertices)
        self.setPos(*center)
        self.contact_points = []
        self.contact_normals = []
        self.mass_center = (0, 0)
        self.bounding_radius = bounding_radius
        self.show_proxy = False
        self.vertex_colors = vertex_colors
        self.alive = True

    def set_children_visibility(self, visibility):
        for child in self.childItems():
            child.setVisible(visibility)

    def paint(self, painter, style, widget=None):
        if not self.show_proxy:
            self.set_children_visibility(True)
            # draw individual triangles
            painter.setPen(QtGui.QPen(Qt.black))
            center = QtCore.QPointF(0,0)

            for p1, p2, vertex_color in zip(self.points+[self.points[:-1]], self.points[1:]+[self.points[0]], self.vertex_colors):
                gradient = QtGui.QLinearGradient(0, 0, p2.x(), p2.y())
                if self.alive:
                    color = QtGui.QColor(*vertex_color[:-1])
                    gradient.setColorAt(0, color.lighter())
                    gradient.setColorAt(1, color)
                else:
                    gradient.setColorAt(0, self.red.lighter())
                    gradient.setColorAt(1, self.red)
                painter.setBrush(QtGui.QBrush(gradient))
                polygon = QtGui.QPolygonF([center, p1, p2])
                painter.drawPolygon(polygon, fillRule=Qt.OddEvenFill)

            painter.setPen(Qt.red)
            painter.setBrush(QtGui.QBrush())

            for contact_normal, contact_point in zip(self.contact_normals, self.contact_points):
                if contact_point is not None and contact_point.any():
                    painter.setPen(Qt.darkGreen)
                    qt_contact_point = QtCore.QPointF(contact_point[0], contact_point[1])
                    painter.drawEllipse(qt_contact_point, 0.5, 0.5)
                    painter.drawLine(qt_contact_point, qt_contact_point+QtCore.QPointF(contact_normal[0], contact_normal[1]))

            painter.setPen(Qt.darkBlue)
            painter.drawEllipse(QtCore.QPointF(self.mass_center[0], self.mass_center[1]), 0.5, 0.5)
        else:
            self.set_children_visibility(False)
            if self.bounding_radius:
                painter.drawEllipse(QtCore.QPointF(0, 0), self.bounding_radius, self.bounding_radius)

    def boundingRect(self):
        return self.shape().boundingRect()

    def shape(self):
        path = QtGui.QPainterPath()
        path.addEllipse(QtCore.QPointF(0, 0), self.bounding_radius, self.bounding_radius)
        return path


class SimulationRangeChooser(QtGui.QWidget):

    step_size_changed = QtCore.pyqtSignal(int)

    def __init__(self, title, initial, start, end, parent=None):
        QtGui.QWidget.__init__(self, parent)

        layout = QtGui.QHBoxLayout()
        self.setLayout(layout)
        layout.addStretch(1)

        label = QtGui.QLabel(title, parent=self)

        line_edit = QtGui.QLineEdit(str(initial), parent=self)
        line_edit.setValidator(QtGui.QIntValidator(start, end, self))

        self.slider = slider = QtGui.QSlider(Qt.Horizontal, parent=self)
        slider.setMinimum(start)
        slider.setMaximum(end)
        slider.setValue(initial)
        slider.valueChanged.connect(lambda: line_edit.setText(str(slider.value())))
        slider.valueChanged.connect(lambda: self.step_size_changed.emit(slider.value()))
        line_edit.returnPressed.connect(lambda: slider.setValue(int(line_edit.text())))
        line_edit.returnPressed.connect(lambda: self.step_size_changed.emit(int(line_edit.text())))

        layout.addWidget(label)
        layout.addWidget(line_edit)
        layout.addWidget(slider)

    def value(self):
        return self.slider.value()


class SettingsDock(QtGui.QDockWidget):

    def __init__(self, genetic_vehicle, parent=None):
        QtGui.QDockWidget.__init__(self, "Settings", parent)
        self.genetic_vehicle = genetic_vehicle
        tab_widget = QtGui.QTabWidget(parent=self)
        tab_widget.setPalette(QtGui.QPalette(Qt.green))
        tab_widget.tabBar().setPalette(QtGui.QPalette(Qt.green))
        tab_widget.setTabPosition(tab_widget.West)

        car_properties = QtGui.QWidget(parent=tab_widget)
        car_layout = QtGui.QVBoxLayout()
        car_properties.setLayout(car_layout)
        tab_widget.addTab(car_properties, "Vehicle")

        self.number_of_cars_slider = SimulationRangeChooser("Number of Vehicles", self.parent().genetic_vehicle.number_of_cars, 1, 32*1024, parent=self)
        self.number_of_cars_slider.step_size_changed.connect(self.handle_number_of_cars_slider)
        car_layout.addWidget(self.number_of_cars_slider)

        self.number_of_vertices_slider = SimulationRangeChooser("Number of Vertices per Vehicle", self.parent().genetic_vehicle.number_of_vertices_per_car, 3, 512, parent=self)
        self.number_of_vertices_slider.step_size_changed.connect(self.handle_number_of_vertices_slider)
        car_layout.addWidget(self.number_of_vertices_slider)

        self.number_of_wheels_slider = SimulationRangeChooser("Number of Wheels per Vehicle", self.parent().genetic_vehicle.number_of_wheels_per_car, 1, 512, parent=self)
        self.number_of_wheels_slider.step_size_changed.connect(self.handle_number_of_wheels_slider)
        car_layout.addWidget(self.number_of_wheels_slider)
        car_layout.addStretch(1)

        simulation_properties = QtGui.QWidget(parent=tab_widget)
        simulation_layout = QtGui.QVBoxLayout()
        simulation_properties.setLayout(simulation_layout)
        tab_widget.addTab(simulation_properties, "Simulation")
        
        self.step_slider = SimulationRangeChooser("Simulation Steps", 1, 1, 500, parent=self)
        self.step_slider.step_size_changed.connect(self.handle_proxy_slider)
        simulation_layout.addWidget(self.step_slider)

        self.constraint_slider = SimulationRangeChooser("Constraint Steps", self.parent().genetic_vehicle.satisfy_constraints, 1, 25, parent=self)
        self.constraint_slider.step_size_changed.connect(self.handle_constraint_slider)
        simulation_layout.addWidget(self.constraint_slider)

        self.delta_slider = SimulationRangeChooser("Delta", int(1/self.parent().genetic_vehicle.delta), 1, 64, parent=self)
        self.delta_slider.step_size_changed.connect(self.handle_delta_slider)
        simulation_layout.addWidget(self.delta_slider)
        simulation_layout.addStretch(1)

        visualization_properties = QtGui.QWidget(parent=tab_widget)
        visualization_layout = QtGui.QVBoxLayout()
        visualization_properties.setLayout(visualization_layout)
        tab_widget.addTab(visualization_properties, "Visualization")
        
        toggle_proxy_mode_button = QtGui.QPushButton("Show Bounding Volumes")
        toggle_proxy_mode_button.setCheckable(True)
        toggle_proxy_mode_button.clicked.connect(self.handle_proxy_button)
        visualization_layout.addWidget(toggle_proxy_mode_button)
        
        self.anti_alias_button = QtGui.QPushButton("Antialiasing")
        self.anti_alias_button.setCheckable(True)
        self.anti_alias_button.clicked.connect(self.handle_anti_aliasing)
        visualization_layout.addWidget(self.anti_alias_button)

        self.histogram_button = QtGui.QPushButton("Show Histogram")
        self.histogram_button.clicked.connect(partial(geneticvehicle.show_histogram, self.parent().genetic_vehicle))
        visualization_layout.addWidget(self.histogram_button)

        self.save_vehicle_images_button = QtGui.QPushButton("Save Vehicle Images")
        self.save_vehicle_images_button.clicked.connect(self.parent().save_vehicle_images)
        visualization_layout.addWidget(self.save_vehicle_images_button)

        self.disable_visualization = QtGui.QPushButton("Disable Visualization")
        self.disable_visualization.setCheckable(True)
        self.disable_visualization.clicked.connect(self.handle_disable_visualization)
        visualization_layout.addWidget(self.disable_visualization)
        visualization_layout.addStretch(1)
        
        genetic_properties = QtGui.QWidget(parent=tab_widget)
        genetic_layout = QtGui.QVBoxLayout()
        genetic_properties.setLayout(genetic_layout)
        tab_widget.addTab(genetic_properties, "Genetic Algorithm")

        self.crossover_button = QtGui.QPushButton("Evolve")
        self.crossover_button.clicked.connect(self.handle_crossover)
        genetic_layout.addWidget(self.crossover_button)
        genetic_layout.addStretch(1)

        self.setWidget(tab_widget)

    def handle_crossover(self):
        if self.parent().genetic_vehicle.evolve():
            self.parent().add_geometry()

    def handle_number_of_cars_slider(self):
        self.parent().genetic_vehicle.number_of_cars = self.number_of_cars_slider.value()
        self.parent().genetic_vehicle.build()
        self.parent().add_geometry()

    def handle_number_of_vertices_slider(self):
        self.parent().genetic_vehicle.number_of_vertices_per_car = self.number_of_vertices_slider.value()
        self.parent().genetic_vehicle.build()
        self.parent().add_geometry()

    def handle_number_of_wheels_slider(self):
        self.parent().genetic_vehicle.number_of_wheels_per_car = self.number_of_wheels_slider.value()
        self.parent().genetic_vehicle.build()
        self.parent().add_geometry()

    def handle_proxy_button(self):
        for car in self.parent().cars:
            car.show_proxy = not car.show_proxy
        self.parent().scene.update()

    def handle_proxy_slider(self):
        self.parent().genetic_vehicle.steps = self.step_slider.value()

    def handle_disable_visualization(self):
        self.parent().disable_visualization = self.disable_visualization.isChecked()

    def handle_delta_slider(self):
        self.parent().genetic_vehicle.delta = 1/self.delta_slider.value()

    def handle_constraint_slider(self):
        self.parent().genetic_vehicle.satisfy_constraints = self.constraint_slider.value()

    def handle_anti_aliasing(self):
        self.parent().graphics_view.setRenderHint(QtGui.QPainter.Antialiasing, on=self.anti_alias_button.isChecked())
        self.parent().graphics_view.setRenderHint(QtGui.QPainter.HighQualityAntialiasing, on=self.anti_alias_button.isChecked())



class Colorama(QtGui.QWidget):

    vehicle_visibility_changed = QtCore.pyqtSignal()

    def __init__(self, genetic_vehicle, main_window, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.genetic_vehicle = genetic_vehicle
        self.main_window = main_window
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.start_vehicle_id = 0

    def mousePressEvent(self, event):
        draw_rect = self.rect()
        spacing = draw_rect.width()/self.genetic_vehicle.number_of_cars
        self.start_vehicle_id = int(event.x()/spacing)

    def mouseReleaseEvent(self, event):
        draw_rect = self.rect()
        spacing = draw_rect.width()/self.genetic_vehicle.number_of_cars
        end_vehicle_id = int(event.x()/spacing)
        start, end = min(self.start_vehicle_id, end_vehicle_id), max(self.start_vehicle_id, end_vehicle_id)
        start = max(0, start)
        end = min(end, self.genetic_vehicle.number_of_cars-1)
        for vehicle_id in range(start, end+1):
            if event.modifiers() == Qt.ControlModifier:
                self.main_window.visible_vehicles[vehicle_id] = False
            elif event.modifiers() == Qt.ShiftModifier:
                self.main_window.visible_vehicles[vehicle_id] = True
            else:
                self.main_window.visible_vehicles[vehicle_id] = not self.main_window.visible_vehicles[vehicle_id]
        self.update()
        self.vehicle_visibility_changed.emit()

    def paintEvent(self, event):
         qp = QtGui.QPainter()
         qp.begin(self)
         draw_rect = self.rect()
         spacing = draw_rect.width()/self.genetic_vehicle.number_of_cars
         qp.setPen(QtGui.QPen(Qt.NoPen))
         #gradient = QtGui.QLinearGradient(0, 0, spacing, 0)
         #gradient.setSpread(gradient.RepeatSpread)
         #gradient.setColorAt(0, Qt.white)
         #gradient.setColorAt(1, Qt.red)
         #qp.setBrush(QtGui.QBrush(gradient))
         for vehicle_id, alive in enumerate(self.genetic_vehicle.vehicle_alive.get()):
            if alive >= 1:
                qp.setBrush(QtGui.QBrush(Qt.green, Qt.SolidPattern))
                if not self.main_window.visible_vehicles[vehicle_id]:
                    qp.setBrush(QtGui.QBrush(Qt.darkGray, Qt.SolidPattern))
            else:
                qp.setBrush(QtGui.QBrush(Qt.red, Qt.SolidPattern))
                if not self.main_window.visible_vehicles[vehicle_id]:
                    qp.setBrush(QtGui.QBrush(Qt.darkRed, Qt.SolidPattern))

            qp.drawRect(QtCore.QRectF(vehicle_id*spacing,0,spacing,draw_rect.height()))
         qp.end()

    def sizeHint(self):
        return QtCore.QSize(900,100)


class ControlDock(QtGui.QDockWidget):

    def __init__(self, genetic_vehicle, parent=None):
        QtGui.QDockWidget.__init__(self, "Controls", parent)
        self.genetic_vehicle = genetic_vehicle

        widget = QtGui.QWidget(parent=self)
        layout = QtGui.QHBoxLayout()
        widget.setLayout(layout)

        self.colorama = Colorama(genetic_vehicle, parent, parent=self)
        layout.addWidget(self.colorama)

        statistics_widget = QtGui.QWidget(parent=self)
        statistics_layout = QtGui.QFormLayout()
        self.alive_percentage = QtGui.QLabel("-")
        statistics_layout.addRow("Alive: ", self.alive_percentage)
        statistics_widget.setLayout(statistics_layout)

        layout.addWidget(statistics_widget)

        self.setWidget(widget)


class InformationDock(QtGui.QDockWidget):

    def __init__(self, genetic_vehicle, parent=None):
        QtGui.QDockWidget.__init__(self, "Information", parent)
        self.genetic_vehicle = genetic_vehicle
        self.table = QtGui.QTableWidget(1,1, parent=self)
        self.table.itemPressed.connect(self.item_pressed)
        self.setWidget(self.table)
        self.update_table()

    def update_table(self):
        self.table.setColumnCount(3)
        self.table.setRowCount(self.genetic_vehicle.number_of_cars)
        self.table.setHorizontalHeaderLabels(["Vehicle ID", "Score", "Alive"])
        scores = self.genetic_vehicle.vehicle_score.get()
        alives = self.genetic_vehicle.vehicle_alive.get()
        for table_id, vehicle_id in enumerate(self.genetic_vehicle.get_sorted_score_ids()):
            score = scores[vehicle_id]
            alive = alives[vehicle_id]
            self.table.setItem(table_id, 0, QtGui.QTableWidgetItem(str(vehicle_id)))
            self.table.setItem(table_id, 1, QtGui.QTableWidgetItem(str(score)))
            self.table.setItem(table_id, 2, QtGui.QTableWidgetItem(str(bool(alive))))

    def item_pressed(self, item):
        vehicle_id = self.genetic_vehicle.get_sorted_score_ids()[item.row()]
        self.parent().graphics_view.fitInView(self.parent().cars[vehicle_id], Qt.KeepAspectRatio)
        self.parent().scaleView(0.125)


class Simulation(QtGui.QMainWindow):

    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setWindowTitle("Genetic Vehicle")
        self.graphics_view = QtGui.QGraphicsView()
        self.graphics_view.wheelEvent = lambda event: event.ignore()
        self.graphics_view.setViewport(QtOpenGL.QGLWidget(QtOpenGL.QGLFormat(QtOpenGL.QGL.SampleBuffers)))
        self.graphics_view.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
        self.graphics_view.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)
        self.graphics_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphics_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setCentralWidget(self.graphics_view)
        self.graphics_view.scale(2, 2)
        self.scene = QtGui.QGraphicsScene()
        self.graphics_view.setScene(self.scene)

        self.genetic_vehicle = GeneticVehicle(number_of_cars=2)
        self.disable_visualization = False

        self.settings_dock = SettingsDock(self.genetic_vehicle, parent=self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.settings_dock)

        self.control_dock = ControlDock(self.genetic_vehicle, parent=self)
        self.control_dock.colorama.vehicle_visibility_changed.connect(self.change_visibility)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.control_dock)

        self.information_dock = InformationDock(self.genetic_vehicle, parent=self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.information_dock)

        self.add_geometry()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(20)
        self.timer.timeout.connect(partial(self.genetic_vehicle.simulation_step, None, self.update_visualization))
        self.timer.start()

    def change_visibility(self):
        for vehicle_index, visibility in enumerate(self.visible_vehicles):
            try:
                self.cars[vehicle_index].setVisible(visibility)
            except IndexError:
                pass

    def resize_to_percentage(self, percentage):
        screen = QtGui.QDesktopWidget().screenGeometry()
        self.resize(screen.width()*percentage/100.0, screen.height()*percentage/100.0)
        scr = QtGui.QApplication.desktop().screenGeometry()
        self.move(scr.center()-self.rect().center())

    def save_vehicle_images(self):

        from PyQt4 import QtSvg
        for index, vehicle in enumerate(self.cars):
            svgGen = QtSvg.QSvgGenerator()
            svgGen.setFileName( "test%s.svg" % index )
            svgGen.setSize(QtCore.QSize(200, 200))
            svgGen.setViewBox(QtCore.QRect(-200, -200, 200, 200))
            svgGen.setTitle("SVG Generator Example Drawing")
            svgGen.setDescription("An SVG drawing created by the SVG Generator "
                                  "Example provided with Qt.")

            painter = QtGui.QPainter()
            painter.begin(svgGen)
            vehicle.paint(painter, QtGui.QStyleOption())
            for child in vehicle.childItems():
                child.transform
                child.paint(painter, QtGui.QStyleOption())
            painter.end()


        """
        for index, vehicle in enumerate(self.cars):
            rect = vehicle.boundingRect()
            print rect.size().toSize()
            vehicle.setScale(20)
            image = QtGui.QImage(rect.size().toSize()*20, QtGui.QImage.Format_RGB32)
            image.fill(QtGui.QColor(255,255,255).rgb())
            painter = QtGui.QPainter(image)
            painter.translate(-rect.topLeft()*20)
            vehicle.paint(painter, QtGui.QStyleOption())
            for child in vehicle.childItems():
                child.paint(painter, QtGui.QStyleOption())
            painter.end()
            image.save("test%s.png" % index)
        """

    def wheelEvent(self, event):
        self.scaleView(math.pow(2.0, -event.delta()/240.0))
        event.accept()
        QtGui.QMainWindow.wheelEvent(self, event)

    def scaleView(self, scaleFactor):
        factor = self.graphics_view.matrix().scale(scaleFactor, scaleFactor).mapRect(QtCore.QRectF(0, 0, 1, 1)).width()
        if factor < 0.07 or factor > 100:
            pass
        self.graphics_view.scale(scaleFactor, scaleFactor)

    def add_geometry(self):
        self.visible_vehicles = numpy.zeros(self.genetic_vehicle.number_of_cars, dtype=numpy.bool)
        self.visible_vehicles += 1

        self.scene.clear()

        self.geometry_lines = zip(self.genetic_vehicle.geometry_points, self.genetic_vehicle.geometry_points[1:])
        for (x1,y1), (x2, y2) in self.geometry_lines:
            self.scene.addItem(QtGui.QGraphicsLineItem(x1, y1, x2, y2))
            radius = 1
            self.scene.addItem(QtGui.QGraphicsEllipseItem(x1-radius/2, y1-radius/2, radius, radius))

        formatted_vehicle_vertices = self.genetic_vehicle.vehicle_vertices.get().reshape((self.genetic_vehicle.number_of_cars, self.genetic_vehicle.number_of_vertices_per_car, 2))
        self.cars = [Car(center, vertices, bounding_radius, vertex_colors) for center, vertices, bounding_radius, vertex_colors in zip(self.genetic_vehicle.vehicle_positions.get(),
                                                                                                                                    formatted_vehicle_vertices,
                                                                                                                                    self.genetic_vehicle.vehicle_bounding_volumes.get(),
                                                                                                                                    grouper(self.genetic_vehicle.number_of_vertices_per_car, self.genetic_vehicle.vertex_colors.get()))]
        self.wheels = [Circle(radius=radius) for radius in self.genetic_vehicle.wheel_radii.get()]

        for car in self.cars:
            self.scene.addItem(car)

        vehicle_vertices = self.genetic_vehicle.vehicle_vertices.get()
        for index, (car, wheel_group, wheel_verts) in enumerate(zip(self.cars,
                                                                    grouper(self.genetic_vehicle.number_of_wheels_per_car, self.wheels),
                                                                    grouper(self.genetic_vehicle.number_of_wheels_per_car, self.genetic_vehicle.wheel_vertex_positions.get()))):
            for wheel, vert_index in zip(wheel_group, wheel_verts):
                x,y = vehicle_vertices[index*self.genetic_vehicle.number_of_vertices_per_car+vert_index]
                wheel.setParentItem(car)
                wheel.setPos(x,y)
                wheel.setOpacity(0.5)

        self.change_visibility()

    def update_visualization(self):
        #self.control_dock.alive_percentage.setText("%s%%" % (100*sum(self.genetic_vehicle.vehicle_alive.get())/self.genetic_vehicle.number_of_cars))
        if not self.disable_visualization or not self.genetic_vehicle.run:
            self.information_dock.update_table()
            self.control_dock.colorama.update()
            if self.cars[0].show_proxy:
                for (x, y), alive, active_item in zip(self.genetic_vehicle.vehicle_positions.get(), self.genetic_vehicle.vehicle_alive.get(), self.cars):
                    active_item.setPos(x, y)
                    # TODO change colors
                    active_item.color = Qt.blue if alive >= 1 else Qt.red

            else:
                for (x,y), orientation, alive, contact_points, contact_normals, (massx, massy), bounding_radius, wheels, wheel_orientations, vertex_color, vehicle in zip(self.genetic_vehicle.vehicle_positions.get(),
                                                                                                                                                            self.genetic_vehicle.vehicle_orientations.get(),
                                                                                                                                                            self.genetic_vehicle.vehicle_alive.get(),
                                                                                                                                                            grouper(self.genetic_vehicle.number_of_contact_points, self.genetic_vehicle.vehicle_contact_points.get()),
                                                                                                                                                            grouper(self.genetic_vehicle.number_of_contact_points, self.genetic_vehicle.vehicle_contact_normals.get()),
                                                                                                                                                            self.genetic_vehicle.vehicle_center_masses.get(),
                                                                                                                                                            self.genetic_vehicle.vehicle_bounding_volumes.get(),
                                                                                                                                                            grouper(self.genetic_vehicle.number_of_wheels_per_car, self.wheels),
                                                                                                                                                            grouper(self.genetic_vehicle.number_of_wheels_per_car, self.genetic_vehicle.wheel_orientations.get()),
                                                                                                                                                            grouper(self.genetic_vehicle.number_of_vertices_per_car, self.genetic_vehicle.vertex_colors.get()),
                                                                                                                                                            self.cars):
                    vehicle.setPos(x, y)
                    vehicle.setRotation(orientation)
                    vehicle.alive = True if alive >= 1 else False
                    vehicle.contact_points = contact_points
                    vehicle.contact_normals = contact_normals
                    vehicle.mass_center = (massx, massy)
                    vehicle.bounding_radius = bounding_radius
                    for wheel_orientation, wheel in zip(wheel_orientations, wheels):
                        wheel.setRotation(wheel_orientation)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F:
            self.genetic_vehicle.process = True
        elif event.key() == Qt.Key_R:
            self.genetic_vehicle.run = not self.genetic_vehicle.run
        elif event.key() == Qt.Key_E:
            self.graphics_view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        elif event.key() == Qt.Key_S:
            vehicle_id = self.genetic_vehicle.get_sorted_score_ids()[0]
            #score = self.genetic_vehicle.vehicle_score.get()[vehicle_id]
            self.graphics_view.fitInView(self.cars[vehicle_id], Qt.KeepAspectRatio)
        elif event.key() == Qt.Key_A:
            self.graphics_view.fitInView(self.cars[0], Qt.KeepAspectRatio)


def display_vehicles(genetic_vehicle, vehicle_ids, width=2*4096, filename="test"):
    app = QtGui.QApplication(sys.argv)
    # get relevant information
    scene = QtGui.QGraphicsScene()
    view = QtGui.QGraphicsView()
    view.setScene(scene)

    accumulated_space = 0
    for index, vehicle_id in enumerate(vehicle_ids):
        center, vertices, bounding_radius, vertex_colors, wheel_radii, wheel_vertex = genetic_vehicle.get_vehicle_information(vehicle_id)
        vehicle = Car((accumulated_space, 0), vertices, bounding_radius, vertex_colors)
        accumulated_space += 2.125*bounding_radius
        for wheel_radius, vert_index in zip(wheel_radii, wheel_vertex):
            x, y = genetic_vehicle.vehicle_vertices.get()[vehicle_id*genetic_vehicle.number_of_vertices_per_car+vert_index]
            wheel = Circle(wheel_radius)
            wheel.setParentItem(vehicle)
            wheel.setPos(x, y)
            wheel.setOpacity(0.5)
        scene.addItem(vehicle)

    #view.fitInView(vehicle, Qt.KeepAspectRatio)
    #view.show()

    rect = scene.sceneRect()
    aspect = rect.height()/rect.width()

    image = QtGui.QImage(width, width*aspect, QtGui.QImage.Format_ARGB32)
    image.fill(QtGui.QColor(255, 255, 255, 0).rgb())
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    scene.render(painter)
    painter.end()
    image.save("results/"+filename + ".png")
    #app.exec_()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    simulation = Simulation()
    simulation.resize_to_percentage(75)
    simulation.show()

    profile = False
    if not profile:
        sys.exit(app.exec_())
    else:
        cProfile.runctx('sys.exit(app.exec_())', {"app" : app, "sys" : sys}, None, 'profdata')
        p = pstats.Stats('profdata')
        p.sort_stats('time').print_stats()