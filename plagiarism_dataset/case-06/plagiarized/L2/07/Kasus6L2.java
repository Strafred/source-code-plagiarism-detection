import java.util.Scanner;
public class Kasus6L2 {
    public static void main (String[] args) {
        Scanner inp = new Scanner(System.in);
        int[] arrNum = new int[10];
        for (int i = 0; i < 10; i++) {
            System.out.print("Read a number: ");
            arrNum[i] = inp.nextInt();
        }
        for (int i = 9; i >= 0; i--) {
            System.out.println(arrNum[i]);
        }
    }
}
