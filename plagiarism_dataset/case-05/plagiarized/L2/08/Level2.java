public class Level2 {
    public static void main(String[] args) {
        System.out.print("Enter an integer: ");
        java.util.Scanner in = new java.util.Scanner(System.in);
        int angka = in.nextInt();
        balik(angka);
    }
    public static void balik(int angka) {
        while (angka != 0) {
            int sisa = angka % 10;
            System.out.print(sisa);
            angka = angka / 10;
        }
        System.out.println();
    }
}
