<?php
$baseDir = 'data';

if (isset($_POST['new_folder'])) {
    $newFolderName = trim($_POST['new_folder']);
    if ($newFolderName !== '') {
        $newPath = $baseDir . '/' . basename($newFolderName);
        if (!file_exists($newPath)) {
            mkdir($newPath);
        }
    }
}

if (isset($_GET['delete'])) {
    $delPath = $baseDir . '/' . basename($_GET['delete']);
    if (is_dir($delPath)) {
        array_map('unlink', glob("$delPath/*.*"));
        rmdir($delPath);
    }
}

if (isset($_POST['rename_folder'])) {
    $oldName = $baseDir . '/' . basename($_POST['old_name']);
    $newName = $baseDir . '/' . basename($_POST['rename_folder']);
    if ($oldName !== $newName && !file_exists($newName)) {
        rename($oldName, $newName);
    }
}

if (isset($_POST['upload_to']) && isset($_FILES['image'])) {
    $targetFolder = $baseDir . '/' . basename($_POST['upload_to']);
    if (is_dir($targetFolder)) {
        $file = $_FILES['image'];
        $targetFile = $targetFolder . '/' . basename($file['name']);
        move_uploaded_file($file['tmp_name'], $targetFile);
    }
}

$folders = array_filter(glob($baseDir . '/*'), 'is_dir');
?>

<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <title>จัดการโฟลเดอร์</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 p-8">
    <div class="max-w-4xl mx-auto bg-white p-6 rounded shadow">
        <h1 class="text-2xl font-bold mb-4">📁 จัดการโฟลเดอร์ใน <code>/data</code></h1>

        <!-- สร้างโฟลเดอร์ -->
        <form method="POST" class="flex gap-2 mb-6">
            <input type="text" name="new_folder" placeholder="ชื่อโฟลเดอร์ใหม่" required
                   class="flex-1 border rounded px-3 py-2">
            <button type="submit"
                    class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">+ สร้างโฟลเดอร์</button>
        </form>

        <div class="space-y-6">
            <?php foreach ($folders as $folder): 
                $name = basename($folder); ?>
                <div class="bg-gray-50 border rounded p-4">
                    <div class="flex justify-between items-center mb-3">
                        <strong class="text-lg"><?= htmlspecialchars($name) ?></strong>

                        <div class="flex gap-2">
                            <!-- ลบ -->
                            <a href="?delete=<?= urlencode($name) ?>"
                               onclick="return confirm('ลบโฟลเดอร์นี้?')"
                               class="text-red-500 hover:underline">ลบ</a>

                            <!-- แก้ไขชื่อ -->
                            <form method="POST" class="flex gap-2">
                                <input type="hidden" name="old_name" value="<?= $name ?>">
                                <input type="text" name="rename_folder" placeholder="ชื่อใหม่" required
                                       class="border px-2 py-1 rounded">
                                <button type="submit" class="bg-yellow-400 px-3 py-1 rounded hover:bg-yellow-500">เปลี่ยนชื่อ</button>
                            </form>
                        </div>
                    </div>

                    <!-- อัปโหลดรูป -->
                    <form method="POST" enctype="multipart/form-data" class="flex items-center gap-2 mb-3">
                        <input type="hidden" name="upload_to" value="<?= $name ?>">
                        <input type="file" name="image" accept="image/*" required class="border rounded px-2 py-1">
                        <button type="submit" class="bg-green-500 text-white px-3 py-1 rounded hover:bg-green-600">
                            อัปโหลดรูป
                        </button>
                    </form>

                    <!-- แสดงรูปภาพ -->
                    <div class="flex flex-wrap gap-2">
                        <?php 
                        $images = glob("$folder/*.{jpg,jpeg,png,gif}", GLOB_BRACE);
                        foreach ($images as $img): ?>
                            <img src="<?= $img ?>" class="w-24 h-24 object-cover rounded border" alt="">
                        <?php endforeach; ?>
                    </div>
                </div>
            <?php endforeach; ?>
        </div>
    </div>
</body>
</html>
