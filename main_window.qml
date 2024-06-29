import QtQuick 6.7
import QtQuick.Controls 6.7


Canvas {
    id: canvas
    width: 600
    height: 400
    antialiasing: true

    onPaint: {
        var ctx = getContext("2d");
        ctx.lineWidth = 3;
        ctx.strokeStyle = "black";
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
			id: quitButton
			text: "Quit"
			onClicked: {
				Qt.quit();
			}
		}

		Button {
			id: saveButton
			text: "Save"
			onClicked: {
				var image = canvas.save("image.png");
			}
		}

	}
}
