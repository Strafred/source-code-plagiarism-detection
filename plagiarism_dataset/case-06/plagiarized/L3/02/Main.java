import java.util.Scanner;
public class Main {
    public static void main (String[] args) {
        int[] number = new int[10];
     Scanner s = new Scanner(System.in);
        for (int i = 0; i < 10; i++) {
            System.out.print("Read a number: ");
            number[i] = s.nextInt();
        }
        for (int i = 9; i >= 0; i--) {
            System.out.println(number[i]);
        }
    }
}
