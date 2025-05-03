<?php 
class Database {
    private $host = 'localhost';
    private $dbname = 'hee';
    private $username = 'root';
    private $password = '1212312121'; // 1212312121
    private $pdo;

    public function __construct() {
        try {
            $this->pdo = new PDO(
                "mysql:host={$this->host};dbname={$this->dbname};charset=utf8",
                $this->username,
                $this->password
            );
            $this->pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        } catch (PDOException $e) {
            die("เชื่อมต่อฐานข้อมูลล้มเหลว: " . $e->getMessage());
        }
    }

    public function getConnection() {
        return $this->pdo;
    }
}
?>