import java.util.Scanner;
public class Kasus5L4 {
    public static void main(String[] args) {
        Scanner inp = new Scanner(System.in);
        System.out.print("Enter an integer: ");
        int angka = inp.nextInt();
        while (angka != 0) {
            int temp = angka % 10;
            angka = angka / 10;
            System.out.print(temp);
        }
        System.out.println();
    }
}
