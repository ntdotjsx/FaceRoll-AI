<?php

include 'class/db.php';

$db = new Database();
$conn = $db->getConnection();

// ดึงข้อมูลโฟลเดอร์ที่มีในฐานข้อมูล
$stmt = $conn->query("SELECT DISTINCT folder_name FROM users");
$usedFolders = $stmt->fetchAll(PDO::FETCH_ASSOC);
$usedFolders = array_map(fn($folder) => $folder['folder_name'], $usedFolders);

// ดึงข้อมูลโฟลเดอร์จากระบบไฟล์
$baseDir = 'data';
$folders = array_filter(glob($baseDir . '/*'), 'is_dir');

// หากมีการส่งฟอร์มเพิ่มผู้ใช้ใหม่
if (isset($_POST['add_user'])) {
    $fname = trim($_POST['fname']);
    $lname = trim($_POST['lname']);
    $folder = trim($_POST['folder']);

    if ($fname !== '' && $lname !== '' && in_array($folder, array_map('basename', $folders)) && !in_array($folder, $usedFolders)) {
        // เพิ่มผู้ใช้ใหม่ในฐานข้อมูล
        $stmt = $conn->prepare("INSERT INTO users (fname, lname, folder_name) VALUES (:fname, :lname, :folder_name)");
        $stmt->bindParam(':fname', $fname);
        $stmt->bindParam(':lname', $lname);
        $stmt->bindParam(':folder_name', $folder);
        $stmt->execute();
    }
}

// การแก้ไขชื่อและนามสกุลผู้ใช้
if (isset($_POST['edit_user'])) {
    $userId = $_POST['user_id'];
    $fname = trim($_POST['edit_fname']);
    $lname = trim($_POST['edit_lname']);

    if ($fname !== '' && $lname !== '') {
        $stmt = $conn->prepare("UPDATE users SET fname = :fname, lname = :lname WHERE id = :id");
        $stmt->bindParam(':fname', $fname);
        $stmt->bindParam(':lname', $lname);
        $stmt->bindParam(':id', $userId);
        $stmt->execute();
    }
}

// ดึงข้อมูลผู้ใช้ที่มีอยู่แล้ว
$stmt = $conn->query("SELECT id, fname, lname, folder_name FROM users");
$users = $stmt->fetchAll(PDO::FETCH_ASSOC);

// การจัดการโฟลเดอร์ต่างๆ ในระบบไฟล์
if (isset($_POST['new_folder'])) {
    $newFolderName = trim($_POST['new_folder']);
    if ($newFolderName !== '') {
        $newPath = $baseDir . '/' . basename($newFolderName);
        if (!file_exists($newPath)) {
            mkdir($newPath);
        }
    }
}

// หากมีการส่งฟอร์มลบผู้ใช้
if (isset($_POST['delete_user'])) {
    $userId = $_POST['delete_user_id'];

    // ลบผู้ใช้จากฐานข้อมูล
    $stmt = $conn->prepare("DELETE FROM users WHERE id = :id");
    $stmt->bindParam(':id', $userId);
    $stmt->execute();
}


// ลบโฟลเดอร์พร้อมไฟล์ภายใน
if (isset($_POST['delete_folder'])) {
    $folderToDelete = basename(trim($_POST['delete_folder']));
    $delPath = $baseDir . '/' . $folderToDelete;

    // เช็คว่าโฟลเดอร์อยู่ภายใต้ baseDir และมีอยู่จริง
    if (is_dir($delPath) && strpos(realpath($delPath), realpath($baseDir)) === 0) {
        // ลบไฟล์ทั้งหมดในโฟลเดอร์
        $files = glob($delPath . '/*');
        foreach ($files as $file) {
            if (is_file($file)) {
                unlink($file);
            }
        }
        // ลบโฟลเดอร์เปล่า
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

if (isset($_POST['delete_image'])) {
    $imageToDelete = $_POST['delete_image'];
    // ป้องกันการลบ path ที่ไม่ใช่ในโฟลเดอร์ data
    if (strpos(realpath($imageToDelete), realpath($baseDir)) === 0 && file_exists($imageToDelete)) {
        unlink($imageToDelete);
    }
}


?>

<!DOCTYPE html>
<html lang="th">

<head>
    <meta charset="UTF-8">
    <title>จัดการโฟลเดอร์</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

</head>

<body class="bg-gray-100">

<div class="modal fade modal-xl border-0" id="hee" tabindex="-1" aria-hidden="true">
                    <div class="modal-dialog modal-dialog-centered modal-xl">
                        <div class="modal-content rounded-lg shadow-lg">
                            <form method="POST" class="space-y-6 p-6">
                                <!-- เพิ่มผู้ใช้ใหม่ -->
                                <h2 class="text-2xl font-bold text-gray-800 mb-4">เพิ่มผู้ใช้ใหม่</h2>
                                <div class="flex flex-col gap-4">
                                    <input type="text" name="fname" placeholder="ชื่อ" required class="border px-4 py-3 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
                                    <input type="text" name="lname" placeholder="นามสกุล" required class="border px-4 py-3 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500">

                                    <select name="folder" required class="border px-4 py-3 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
                                        <option value="">เลือกโฟลเดอร์</option>
                                        <?php foreach ($folders as $folder): ?>
                                            <?php $folderName = basename($folder); ?>
                                            <?php if (!in_array($folderName, $usedFolders)): ?>
                                                <option value="<?= $folderName ?>"><?= $folderName ?></option>
                                            <?php endif; ?>
                                        <?php endforeach; ?>
                                    </select>
                                </div>

                                <button type="submit" name="add_user" class="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition duration-300 w-full">เพิ่มผู้ใช้</button>
                            </form>

                            <!-- สร้างโฟลเดอร์ -->
                            <form method="POST" class="flex gap-3 mb-3 p-6 rounded-lg">
                                <input type="text" name="new_folder" placeholder="ชื่อโฟลเดอร์ใหม่" required class="flex-1 border rounded-lg px-4 py-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
                                <button type="submit" class="bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 transition duration-300">+ สร้างโฟลเดอร์</button>
                            </form>

                            <!-- ลบโฟลเดอร์ -->
                            <form method="POST" class="flex gap-3 mb-3 p-6 rounded-lg" onsubmit="return confirm('แน่ใจหรือไม่ว่าต้องการลบโฟลเดอร์นี้?');">
                                <input type="text" name="delete_folder" placeholder="ชื่อโฟลเดอร์ที่จะลบ" required class="flex-1 border rounded-lg px-4 py-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-red-500">
                                <button type="submit" class="bg-red-600 text-white px-6 py-3 rounded-lg hover:bg-red-700 transition duration-300">🗑️ ลบโฟลเดอร์</button>
                            </form>
                        </div>
                    </div>
                </div>

    <!-- ✅ เนื้อหาหลัก -->
    <div class="max-w-4xl mx-auto bg-white p-6 mt-8 rounded shadow">
        <h1 class="text-2xl font-bold mb-4 d-flex justify-between items-center">
            <span>
                📁 DATA TRAIN AI <code>/data</code>
            </span>
            <button type="button" class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#hee">
                FOLDER
            </button>
        </h1>
        


        <!-- ฟอร์มเพิ่มผู้ใช้ -->

        <div class="space-y-6">
            <?php foreach ($users as $user): ?>
                <div class="bg-gray-50 border rounded p-3 mb-4 d-flex justify-content-between align-items-center">
                    <?php
                    $folderName = $user['folder_name'];
                    $folderPath = $baseDir . '/' . $folderName;
                    $imageFiles = glob("$folderPath/*.{jpg,jpeg,png,gif}", GLOB_BRACE);

                    if (count($imageFiles) > 0) {
                        $firstImage = $imageFiles[0]; // เลือกรูปภาพแรกจากโฟลเดอร์
                    ?>
                        <img src="<?= $firstImage ?>" alt="รูปตัวอย่าง" class="w-16 h-16 object-cover rounded-lg border">
                    <?php } else { ?>
                        <span class="text-gray-400">ไม่มีรูป</span>
                    <?php } ?>
                    <p class="font-semibold mb-0">
                        <?= $user['fname'] ?> <?= $user['lname'] ?> -
                        <span class="text-gray-500"><?= $user['folder_name'] ?></span>
                    </p>

                    <!-- ปุ่มเปิด modal -->
                    <button type="button" class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#manageUserModal<?= $user['id'] ?>">
                        จัดการ
                    </button>
                </div>

                <!-- Modal -->
                <div class="modal fade modal-xl border-0" id="manageUserModal<?= $user['id'] ?>" tabindex="-1" aria-labelledby="modalLabel<?= $user['id'] ?>" aria-hidden="true">
                    <div class="modal-dialog modal-dialog-centered modal-xl">
                        <div class="modal-content rounded-lg shadow-lg">
                            <div class="modal-header bg-primary text-white">
                                <h5 class="modal-title" id="modalLabel<?= $user['id'] ?>">จัดการผู้ใช้: <?= $user['fname'] ?> <?= $user['lname'] ?></h5>
                                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="ปิด"></button>
                            </div>

                            <div class="modal-body">
                                <div class="row g-4">
                                    <!-- ฝั่งซ้าย: ฟอร์มจัดการ -->
                                    <div class="col-md-6">
                                        <form method="POST" class="space-y-3">
                                            <input type="hidden" name="user_id" value="<?= $user['id'] ?>">
                                            <div class="mb-3">
                                                <label class="form-label">ชื่อ</label>
                                                <input type="text" name="edit_fname" value="<?= $user['fname'] ?>" class="form-control" required>
                                            </div>
                                            <div class="mb-3">
                                                <label class="form-label">นามสกุล</label>
                                                <input type="text" name="edit_lname" value="<?= $user['lname'] ?>" class="form-control" required>
                                            </div>
                                            <button type="submit" name="edit_user" class="btn btn-warning w-100">✏️ อัปเดตข้อมูล</button>
                                        </form>

                                        <hr class="my-4">

                                        <form method="POST" onsubmit="return confirm('คุณแน่ใจหรือไม่ที่จะลบผู้ใช้นี้?');">
                                            <input type="hidden" name="delete_user_id" value="<?= $user['id'] ?>">
                                            <button type="submit" name="delete_user" class="btn btn-danger w-100">🗑️ ลบผู้ใช้</button>
                                        </form>
                                    </div>

                                    <!-- ฝั่งขวา: แสดงและอัปโหลดภาพ -->
                                    <div class="col-md-6">
                                        <?php
                                        $folderName = $user['folder_name'];
                                        $folderPath = $baseDir . '/' . $folderName;
                                        if (is_dir($folderPath)):
                                        ?>
                                            <div class="bg-light border rounded p-3">
                                                <h5 class="mb-3">📁 โฟลเดอร์: <?= $folderName ?></h5>

                                                <!-- ฟอร์มอัปโหลด -->
                                                <form method="POST" enctype="multipart/form-data" class="d-flex align-items-center gap-2 mb-3">
                                                    <input type="hidden" name="upload_to" value="<?= $folderName ?>">
                                                    <input type="file" name="image" accept="image/*" required class="form-control">
                                                    <button type="submit" class="btn btn-primary">
                                                        ⬆️
                                                    </button>
                                                </form>

                                                <!-- รูปภาพ -->
                                                <div class="d-flex flex-wrap gap-2">
                                                    <?php
                                                    $images = glob("$folderPath/*.{jpg,jpeg,png,gif}", GLOB_BRACE);
                                                    foreach ($images as $img): ?>
                                                        <div class="position-relative" style="width: 80px; height: 80px;">
                                                            <img src="<?= $img ?>" class="rounded border w-100 h-100" style="object-fit: cover;" alt="">
                                                            <form method="POST" onsubmit="return confirm('ลบรูปภาพนี้แน่ใจใช่ไหม?');" class="position-absolute top-0 end-0">
                                                                <input type="hidden" name="delete_image" value="<?= $img ?>">
                                                                <button type="submit" class="btn btn-sm btn-danger p-1" style="font-size: 0.75rem; line-height: 1;">
                                                                    ✖
                                                                </button>
                                                            </form>
                                                        </div>
                                                    <?php endforeach; ?>

                                                </div>
                                            </div>
                                        <?php endif; ?>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

            <?php endforeach; ?>

        </div>
    </div>
</body>

</html>