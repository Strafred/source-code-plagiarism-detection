import java.util.Scanner;
public class Kasus6L6 {
    public static void main(String[] args) {
        int[] arrNum = new int[10];
        Scanner inp = new Scanner(System.in);
        int i = 9;
        while(i >= 0){
            System.out.print("Read a number: ");
            arrNum[i] = inp.nextInt();
            i--;
        }
        Reverse(arrNum);
    }
    private static void Reverse(int[] arr) {
        int i = 0;
        while (i < 10) {
            System.out.println(arr[i]);
            i++;
        }
    }
}
