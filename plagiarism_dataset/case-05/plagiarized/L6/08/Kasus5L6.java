import java.util.Scanner;
public class Kasus5L6 {
    public static void main(String[] args) {
        Scanner inp = new Scanner(System.in);
        System.out.print("Enter an integer: ");
        String angka = inp.next();
        for (int i = angka.length() - 1; i >= 0; i--) {
            System.out.print(angka.charAt(i));
        }
        System.out.println();
    }
}
