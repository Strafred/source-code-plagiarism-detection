public class T04 {
    public static void main(String[] args){
        System.out.println("Miles\tKilometers");
        for(int i = 1; i <= 10; i++){
            System.out.printf(i + "\t%.3f", (i * 1.609));
            System.out.println();
        }
    }
}
