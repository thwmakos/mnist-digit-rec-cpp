import QtQuick 6.0
import QtQuick.Controls 6.0


Canvas {
    id: canvas
    width: 600
    height: 400
    antialiasing: true

    onPaint: {
        var ctx = getContext("2d");

		// fill background with black
		ctx.fillStyle = "black";
		ctx.fillRect(0, 0, width, height);
		
		// drawing color is white
        ctx.lineWidth = 3;
        ctx.strokeStyle = "white";

		// 
        ctx.beginPath();
        for (var i = 0; i < points.length - 1; i++) {
            ctx.moveTo(points[i].x, points[i].y);
            ctx.lineTo(points[i + 1].x, points[i + 1].y);
        }
        ctx.stroke();
    }

    MouseArea {
        anchors.fill: parent
        hoverEnabled: true

        onPressed: (mouse) => {
            canvas.points.push(Qt.point(mouse.x, mouse.y));
        }

        onPositionChanged: (mouse) => {
            if (mouse.buttons & Qt.LeftButton) {
                canvas.points.push(Qt.point(mouse.x, mouse.y));
                canvas.requestPaint();
            }
        }
    }

    property var points: []
	
	Row {
		spacing: 5
		
		Button {
			id: button_clear
			text: "Clear"
			onClicked: {
				canvas.points = [];
				canvas.requestPaint();
			}	
		}

		Button {
			id: button_save
			text: "Save"
			onClicked: {
				var image = canvas.save("image.png");
			}
		}
		
		Button {
			id: button_quit
			text: "Quit"
			onClicked: {
				Qt.quit();
			}
		}

	}
}
