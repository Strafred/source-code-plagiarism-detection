import java.util.Scanner;
public class Kasus5L2 {
    public static void main(String[] args) {
        System.out.print("Enter an integer: ");
        Scanner inp = new Scanner(System.in);
        int angka = inp.nextInt();
        reverseNumber(angka);
    }
    public static void reverseNumber(int angka) {
        while (angka != 0) {
            int temp = angka % 10;
            System.out.print(temp);
            angka = angka / 10;
        }
        System.out.println();
    }
}
