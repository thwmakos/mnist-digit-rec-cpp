//
// ~thwmakos~
//
// Sat 29 Jun 21:12:05 BST 2024
//
// gui_main.cpp
//
//
// main() function for the GUI build of mnist-digit-rec-cpp
// uses a simple Qt QML interface to allow drawing
// and then recognising the digit based on the network
// implemented in the other files

#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QPushButton>
#include <QMouseEvent>
#include <QPainter>
#include <QFileDialog>

class DrawingWidget : public QWidget 
{
	public:
		DrawingWidget(QWidget *parent = nullptr) : 
			QWidget(parent), 
			m_drawing(false) 
		{
			setFixedSize(640, 480);
			m_image = QImage(size(), QImage::Format_RGB32);
			m_image.fill(Qt::white);
		}

	protected:
		void paintEvent(QPaintEvent *) override 
		{
			QPainter painter(this);
			painter.drawImage(rect(), m_image);
		}

		void mousePressEvent(QMouseEvent *event) override 
		{
			if(event->button() == Qt::LeftButton) 
			{
				m_drawing = true;
				m_last_point = event->pos();
			}
		}

    void mouseMoveEvent(QMouseEvent *event) override 
	{
        if((event->buttons() & Qt::LeftButton) && m_drawing) 
		{
            QPainter painter(&m_image);
            painter.setPen(QPen(Qt::black, 2, Qt::SolidLine, Qt::RoundCap));
            painter.drawLine(m_last_point, event->pos());
            m_last_point = event->pos();
            update();
        }
    }

    void mouseReleaseEvent(QMouseEvent *event) override 
	{
        if (event->button() == Qt::LeftButton) 
		{
            m_drawing = false;
        }
    }

	public:
		void save_scaled_image() 
		{
			QString filename = QFileDialog::getSaveFileName(this, "Save Image", "", "PNG (*.png)");
			
			if (!filename.isEmpty()) 
			{
				QImage scaledImage = m_image.scaled(28, 28, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
				QImage grayscaleImage = scaledImage.convertToFormat(QImage::Format_Grayscale8);
				grayscaleImage.save(filename, "png");
			}
		}

	private:
		QImage m_image;
		QPoint m_last_point;
		bool   m_drawing;
	};

class MainWindow : public QMainWindow 
{
	public:
		MainWindow(QWidget *parent = nullptr) : QMainWindow(parent) 
		{
			QWidget *central_widget = new QWidget(this);
			QVBoxLayout *layout = new QVBoxLayout(central_widget);

			m_drawing_widget = new DrawingWidget(this);
			layout->addWidget(m_drawing_widget);

			QPushButton *save_button = new QPushButton("Save as 28x28 Grayscale", this);
			connect(save_button, &QPushButton::clicked, m_drawing_widget, &DrawingWidget::save_scaled_image);
			layout->addWidget(save_button);

			setCentralWidget(central_widget);
			setGeometry(100, 100, 640, 520);
			setWindowTitle("mnist-digit-rec-cpp gui");
		}

	private:
		DrawingWidget *m_drawing_widget;
};

int main(int argc, char *argv[]) 
{
    QApplication app(argc, argv);
    MainWindow window;
    window.show();
    return app.exec();
}

