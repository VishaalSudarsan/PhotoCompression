diff --git a//dev/null b/read_inputphoto.py
index 0000000000000000000000000000000000000000..da9c8592f54c2516e1ccbd34a4c610411d51167c 100644
--- a//dev/null
+++ b/read_inputphoto.py
@@ -0,0 +1,35 @@
+#!/usr/bin/env python3
+"""Utility to inspect inputphoto and convert it to a pixel matrix."""
+
+from pathlib import Path
+from typing import List
+
+from PIL import Image
+
+
+def main() -> None:
+    path = Path("inputphoto")
+    data = path.read_bytes()
+    print(f"Read {len(data)} bytes from {path.name}")
+    preview = " ".join(f"{byte:02x}" for byte in data[:20])
+    print(f"First 20 bytes: {preview}")
+
+    with Image.open(path) as img:
+        img = img.convert("L")
+        width, height = img.size
+        if width != height:
+            raise ValueError(
+                f"Input photo must be square; got {width}x{height}"
+            )
+        pixels = list(img.getdata())
+
+    matrix: List[List[int]] = [
+        pixels[i * width : (i + 1) * width] for i in range(height)
+    ]
+    print(f"Image converted to {width}x{height} grayscale matrix")
+    top_left = [row[:5] for row in matrix[:5]]
+    print(f"Top-left 5x5 block: {top_left}")
+
+
+if __name__ == "__main__":
+    main()
