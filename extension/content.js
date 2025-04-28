(function() {
  // Create overlay
  const overlay = document.createElement('div');
  overlay.id = 'captcha-selector-overlay';
  Object.assign(overlay.style, {
    position: 'fixed',
    top: '0',
    left: '0',
    width: '100%',
    height: '100%',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    zIndex: '999999',
    cursor: 'crosshair'
  });
  document.body.appendChild(overlay);

  // Instruction text
  const instruction = document.createElement('div');
  instruction.innerText = 'Drag to select CAPTCHA area. Press Esc to cancel.';
  Object.assign(instruction.style, {
    position: 'absolute',
    top: '10px',
    left: '10px',
    color: 'white',
    fontSize: '20px',
    zIndex: '1000000'
  });
  overlay.appendChild(instruction);

  // Selection box element
  const selectionBox = document.createElement('div');
  Object.assign(selectionBox.style, {
    border: '2px dashed #fff',
    position: 'absolute'
  });
  overlay.appendChild(selectionBox);

  let startX, startY, isDragging = false;

  // Mouse down: start selection
  overlay.addEventListener('mousedown', (e) => {
    isDragging = true;
    startX = e.clientX;
    startY = e.clientY;
    selectionBox.style.left = startX + 'px';
    selectionBox.style.top = startY + 'px';
    selectionBox.style.width = '0px';
    selectionBox.style.height = '0px';
  });

  // Mouse move: update selection box
  overlay.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const currentX = e.clientX;
    const currentY = e.clientY;
    const x = Math.min(startX, currentX);
    const y = Math.min(startY, currentY);
    const width = Math.abs(startX - currentX);
    const height = Math.abs(startY - currentY);
    selectionBox.style.left = x + 'px';
    selectionBox.style.top = y + 'px';
    selectionBox.style.width = width + 'px';
    selectionBox.style.height = height + 'px';
  });

  // Mouse up: finalize selection and process it
  overlay.addEventListener('mouseup', () => {
    if (!isDragging) return;
    isDragging = false;
    const rect = selectionBox.getBoundingClientRect();
    if (rect.width < 5 || rect.height < 5) {
      instruction.innerText = 'Selection too small. Press Esc to cancel.';
      return;
    }
    // Prevent further pointer events
    overlay.style.pointerEvents = 'none';
    instruction.innerText = 'Processing selection...';

    // Request screenshot from background
    chrome.runtime.sendMessage({ type: 'captureScreenshot' }, async (response) => {
      const dataUrl = response.dataUrl;
      try {
        const blob = await cropImage(dataUrl, {
          x: rect.left,
          y: rect.top,
          width: rect.width,
          height: rect.height
        });
        // Convert blob to data URL
        const reader = new FileReader();
        reader.onload = function() {
          const croppedDataUrl = reader.result;
          console.log("Cropped data URL ready.");
          // Send the cropped image data URL to background for upload
          chrome.runtime.sendMessage({ type: 'uploadImage', dataUrl: croppedDataUrl }, (resp) => {
            console.log("Received background response:", resp);
            if (chrome.runtime.lastError) {
              console.error("Runtime error:", chrome.runtime.lastError);
            }
            if (resp && resp.success) {
              const resultData = {
                croppedDataUrl: croppedDataUrl,
                predictions: resp.data.predictions,
                captcha_image: resp.data.captcha_image
              };
              chrome.storage.local.set({ resultData: resultData }, () => {
                console.log("Result data saved. Requesting background to open result popup.");
                overlay.remove();
                chrome.runtime.sendMessage({ type: "openResultPopup" });
              });
            } else {
              instruction.innerText = 'Error: ' + (resp && resp.error ? resp.error : 'Upload failed.');
            }
          });
        };
        reader.readAsDataURL(blob);
      } catch (error) {
        instruction.innerText = 'Error: ' + error.message;
      }
    });
  });

  // Cancel overlay on Esc key
  document.addEventListener('keydown', function escHandler(e) {
    if (e.key === 'Escape') {
      overlay.remove();
      document.removeEventListener('keydown', escHandler);
    }
  });

  // Crop the image from the screenshot to the selected rectangle.
  // This function scales the coordinates based on the screenshot's natural size.
  function cropImage(dataUrl, rect) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        const scaleX = image.naturalWidth / window.innerWidth;
        const scaleY = image.naturalHeight / window.innerHeight;
        const sx = rect.x * scaleX;
        const sy = rect.y * scaleY;
        const sw = rect.width * scaleX;
        const sh = rect.height * scaleY;
        const canvas = document.createElement('canvas');
        canvas.width = sw;
        canvas.height = sh;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, sx, sy, sw, sh, 0, 0, sw, sh);
        canvas.toBlob((blob) => {
          if (blob) resolve(blob);
          else reject(new Error('Canvas is empty'));
        }, 'image/png');
      };
      image.onerror = reject;
      image.src = dataUrl;
    });
  }
})();
